#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <LittleFS.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <mbedtls/md.h>
#include <math.h>
#include "driver/gpio.h"

// ==================== MODE FLAGS ====================
#define USE_MOCK_SENSORS true   // Set false when INA219 + BME280 are wired
#define BENCH_TEST_MODE  true   // Set false for 5-min deep sleep field deployment

#if !USE_MOCK_SENSORS
  #include <Adafruit_INA219.h>
  #include <Adafruit_BME280.h>
  Adafruit_INA219 ina_bat(0x40);
  Adafruit_INA219 ina_uc(0x41);
  Adafruit_INA219 ina_pv(0x44);
  Adafruit_BME280 bme;
#endif

// ==================== USER CONFIGURATION ====================
const char* WIFI_SSID  = "YOUR_WIFI_SSID";
const char* WIFI_PASS  = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "https://your-app.fastapicloud.dev"; // FastAPI Cloud URL (no trailing slash)
const char* NODE_ID    = "esp32_whitehouse_01";

// GPIO Switch Pins
constexpr int PIN_SW_DIS_BAT = 25; // Normally-ON (BC547+BC557): LOW = ON, HIGH = OFF
constexpr int PIN_SW_DIS_UC  = 26; // Normally-OFF (BC547): HIGH = ON, LOW = OFF
constexpr int PIN_SW_CHG_BAT = 27; // Normally-OFF (BC547): HIGH = ON, LOW = OFF
constexpr int PIN_SW_CHG_UC  = 14; // Normally-OFF (BC547): HIGH = ON, LOW = OFF

// Storage Safety Limits (3S2P Li-ion + 2S 3.0V 3000F Ultracap = 1500F effective)
constexpr float V_BAT_MIN    = 9.6f;   // ~20% SoC floor (3S2P)
constexpr float V_BAT_MAX    = 12.3f;  // Longevity CV ceiling (4.10V/cell)
constexpr float V_UC_MIN     = 2.5f;   // Minimum usable input for MT3608 boost stage
constexpr float V_UC_MAX     = 5.75f;  // Safe ceiling for 2S 3.0V (6.0V max) bank
constexpr float V_PV_BAT_MIN = 13.0f;  // Min PV voltage to charge 3S2P effectively
constexpr float TEMP_MAX_C   = 45.0f;  // Enclosure battery charging cutoff

// Static PSU Metadata: [Bat Cap norm, UC Cap norm (1500F/3000F), Solar W norm (40W/100W)]
constexpr float STAT_META[3] = { 0.52f, 0.50f, 0.40f };

// Timing & FL Parameters
#if BENCH_TEST_MODE
  constexpr uint32_t WAKE_INTERVAL_SEC = 10;  // 10s quick bench cycle
  constexpr uint32_t FL_SYNC_EVERY     = 3;   // Sync every 3 wake cycles on bench
#else
  constexpr uint32_t WAKE_INTERVAL_SEC = 300; // 5 minutes field cycle
  constexpr uint32_t FL_SYNC_EVERY     = 12;  // Hourly/daytime sync check
#endif

constexpr int   NUM_WEIGHTS     = 375;
constexpr int   MAX_LOG_SAMPLES = 288;
constexpr int   LOCAL_EPOCHS    = 5;
constexpr float LEARNING_RATE   = 0.01f;
constexpr float FEDPROX_MU      = 0.01f;
constexpr int   MIN_SAMPLES_FL  = 6;

// ==================== RTC MEMORY (Survives Deep Sleep) ====================
RTC_DATA_ATTR uint32_t boot_count       = 0;
RTC_DATA_ATTR int      current_round    = -1;
RTC_DATA_ATTR float    last_tx_temp     = 28.0f;
RTC_DATA_ATTR float    last_tx_hum      = 65.0f;
RTC_DATA_ATTR float    last_tx_pres     = 1010.0f;
RTC_DATA_ATTR float    last_tx_irr      = 0.0f;
RTC_DATA_ATTR uint32_t mins_since_tx    = 999;
RTC_DATA_ATTR float    active_dt_tx_min = 30.0f;
RTC_DATA_ATTR float    active_delta     = 0.25f;
RTC_DATA_ATTR float    active_pi_chg    = 0.50f;
RTC_DATA_ATTR uint32_t last_fl_boot     = 0;

// ==================== 375-PARAM MIXED-INPUT MLP ====================
struct MixedInputMLP {
  float w[NUM_WEIGHTS];
  float w_global[NUM_WEIGHTS];
  float h_dyn[12];
  float h_stat[6];
  float h_cat[18];
  float h_fus[12];
  float out[3];

  static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
  static inline float sigmoid(float x) {
    x = constrain(x, -15.0f, 15.0f);
    return 1.0f / (1.0f + expf(-x));
  }

  void initRandom() {
    for (int i = 0; i < NUM_WEIGHTS; i++) {
      w[i] = ((float)random(-100, 101)) / 1000.0f;
      w_global[i] = w[i];
    }
  }

  void forward(const float x_dyn[6], const float x_stat[3]) {
    int idx = 0;
    for (int j = 0; j < 12; j++) {
      float sum = w[idx + 72 + j];
      for (int i = 0; i < 6; i++) sum += x_dyn[i] * w[idx + i * 12 + j];
      h_dyn[j] = relu(sum);
      h_cat[j] = h_dyn[j];
    }
    idx += 84;

    for (int j = 0; j < 6; j++) {
      float sum = w[idx + 18 + j];
      for (int i = 0; i < 3; i++) sum += x_stat[i] * w[idx + i * 6 + j];
      h_stat[j] = relu(sum);
      h_cat[12 + j] = h_stat[j];
    }
    idx += 24;

    for (int j = 0; j < 12; j++) {
      float sum = w[idx + 216 + j];
      for (int i = 0; i < 18; i++) sum += h_cat[i] * w[idx + i * 12 + j];
      h_fus[j] = relu(sum);
    }
    idx += 228;

    for (int k = 0; k < 3; k++) {
      float sum = w[idx + 36 + k];
      for (int j = 0; j < 12; j++) sum += h_fus[j] * w[idx + j * 3 + k];
      out[k] = sigmoid(sum);
    }
  }

  float trainStep(const float x_dyn[6], const float x_stat[3], const float y_true[3], float lr, float mu) {
    forward(x_dyn, x_stat);

    float err0 = out[0] - y_true[0];
    float err1 = out[1] - y_true[1];
    float eps  = 1e-6f;
    float loss = 0.5f * (err0 * err0 + err1 * err1)
               - (y_true[2] * logf(out[2] + eps) + (1.0f - y_true[2]) * logf(1.0f - out[2] + eps));

    float d_out[3];
    d_out[0] = err0 * out[0] * (1.0f - out[0]);
    d_out[1] = err1 * out[1] * (1.0f - out[1]);
    d_out[2] = out[2] - y_true[2];

    float grads[NUM_WEIGHTS] = {0};
    const int off_dyn  = 0;
    const int off_stat = 84;
    const int off_fus  = 108;
    const int off_head = 336;

    float d_fus[12] = {0};
    for (int k = 0; k < 3; k++) {
      grads[off_head + 36 + k] = d_out[k];
      for (int j = 0; j < 12; j++) {
        grads[off_head + j * 3 + k] = h_fus[j] * d_out[k];
        d_fus[j] += w[off_head + j * 3 + k] * d_out[k];
      }
    }
    for (int j = 0; j < 12; j++) {
      if (h_fus[j] <= 0.0f) d_fus[j] = 0.0f;
    }

    float d_cat[18] = {0};
    for (int j = 0; j < 12; j++) {
      grads[off_fus + 216 + j] = d_fus[j];
      for (int i = 0; i < 18; i++) {
        grads[off_fus + i * 12 + j] = h_cat[i] * d_fus[j];
        d_cat[i] += w[off_fus + i * 12 + j] * d_fus[j];
      }
    }

    for (int j = 0; j < 12; j++) {
      float dg = (h_dyn[j] > 0.0f) ? d_cat[j] : 0.0f;
      grads[off_dyn + 72 + j] = dg;
      for (int i = 0; i < 6; i++) grads[off_dyn + i * 12 + j] = x_dyn[i] * dg;
    }

    for (int j = 0; j < 6; j++) {
      float dg = (h_stat[j] > 0.0f) ? d_cat[12 + j] : 0.0f;
      grads[off_stat + 18 + j] = dg;
      for (int i = 0; i < 3; i++) grads[off_stat + i * 6 + j] = x_stat[i] * dg;
    }

    for (int p = 0; p < NUM_WEIGHTS; p++) {
      float g = grads[p] + mu * (w[p] - w_global[p]);
      g = constrain(g, -5.0f, 5.0f);
      w[p] -= lr * g;
    }
    return loss;
  }
};

MixedInputMLP mlp;

// ==================== LOGGING & STORAGE ====================
struct SampleRecord {
  float x_dyn[6];
  float y_target[3];
};

struct SensorSnapshot {
  float temp, hum, pres, irr, enc_temp;
  float v_bat, i_bat, bat_soc;
  float v_uc, i_uc;
  float v_pv, i_pv, p_pv;
};

void computeSha256Hex(const float* weights, int count, char out_hex[65]) {
  uint8_t hash[32];
  mbedtls_md_context_t ctx;
  mbedtls_md_init(&ctx);
  mbedtls_md_setup(&ctx, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 0);
  mbedtls_md_starts(&ctx);
  mbedtls_md_update(&ctx, (const uint8_t*)weights, count * sizeof(float));
  mbedtls_md_finish(&ctx, hash);
  mbedtls_md_free(&ctx);

  for (int i = 0; i < 32; i++) sprintf(out_hex + i * 2, "%02x", hash[i]);
  out_hex[64] = '\0';
}

bool loadWeightsFromFlash() {
  if (!LittleFS.exists("/weights.bin")) return false;
  File f = LittleFS.open("/weights.bin", "r");
  if (!f || f.size() != sizeof(mlp.w)) {
    if (f) f.close();
    return false;
  }
  f.read((uint8_t*)mlp.w, sizeof(mlp.w));
  f.close();
  memcpy(mlp.w_global, mlp.w, sizeof(mlp.w));
  return true;
}

void saveWeightsToFlash() {
  File f = LittleFS.open("/weights.bin", "w");
  if (f) {
    f.write((uint8_t*)mlp.w, sizeof(mlp.w));
    f.close();
  }
}

int getLoggedSampleCount() {
  if (!LittleFS.exists("/logs.bin")) return 0;
  File f = LittleFS.open("/logs.bin", "r");
  int count = f ? (f.size() / sizeof(SampleRecord)) : 0;
  if (f) f.close();
  return count;
}

void appendLogSample(const SampleRecord& rec) {
  if (getLoggedSampleCount() >= MAX_LOG_SAMPLES) {
    LittleFS.remove("/logs.bin");
  }
  File f = LittleFS.open("/logs.bin", "a");
  if (f) {
    f.write((const uint8_t*)&rec, sizeof(SampleRecord));
    f.close();
  }
}

// ==================== HARDWARE SWITCHING & SENSING ====================
void initSwitchPins() {
  gpio_hold_dis((gpio_num_t)PIN_SW_CHG_BAT);
  gpio_hold_dis((gpio_num_t)PIN_SW_CHG_UC);

  pinMode(PIN_SW_DIS_BAT, OUTPUT);
  pinMode(PIN_SW_DIS_UC,  OUTPUT);
  pinMode(PIN_SW_CHG_BAT, OUTPUT);
  pinMode(PIN_SW_CHG_UC,  OUTPUT);

  digitalWrite(PIN_SW_DIS_BAT, LOW); // Normally-ON Battery Base
  digitalWrite(PIN_SW_DIS_UC,  LOW); // Ultracap OFF
  digitalWrite(PIN_SW_CHG_BAT, LOW);
  digitalWrite(PIN_SW_CHG_UC,  LOW);
}

bool switchToUltracapTX(float v_uc) {
  if (v_uc < V_UC_MIN) return false;
  digitalWrite(PIN_SW_DIS_UC, HIGH);  // Make Ultracap first
  delay(5);
  digitalWrite(PIN_SW_DIS_BAT, HIGH); // Break Battery second
  return true;
}

void switchToBatteryBase() {
  digitalWrite(PIN_SW_DIS_BAT, LOW);  // Make Battery first
  delay(5);
  digitalWrite(PIN_SW_DIS_UC, LOW);   // Break Ultracap second
}

void manageSolarCharging(const SensorSnapshot& s, float pi_chg) {
  digitalWrite(PIN_SW_CHG_BAT, LOW);
  digitalWrite(PIN_SW_CHG_UC,  LOW);

  if (s.p_pv < 0.2f) return;

  bool uc_needs_charge  = (s.v_uc < V_UC_MAX);
  bool bat_needs_charge = (s.v_bat < V_BAT_MAX) && (s.enc_temp < TEMP_MAX_C);

  if (s.v_pv < V_PV_BAT_MIN && uc_needs_charge) {
    digitalWrite(PIN_SW_CHG_UC, HIGH);
  } else if (pi_chg > 0.5f && uc_needs_charge) {
    digitalWrite(PIN_SW_CHG_UC, HIGH);
  } else if (bat_needs_charge) {
    digitalWrite(PIN_SW_CHG_BAT, HIGH);
  } else if (uc_needs_charge) {
    digitalWrite(PIN_SW_CHG_UC, HIGH);
  }
}

SensorSnapshot readAllSensors() {
  SensorSnapshot s = {};
#if USE_MOCK_SENSORS
  float hour = fmodf((boot_count * 5.0f) / 60.0f, 24.0f);
  s.temp     = 26.0f + 5.0f * sinf(hour * 0.26f);
  s.hum      = 70.0f - 15.0f * sinf(hour * 0.26f);
  s.pres     = 1012.0f;
  s.irr      = (hour > 6.5f && hour < 18.0f) ? 650.0f * sinf((hour - 6.5f) * 3.1416f / 11.5f) : 0.0f;
  s.enc_temp = s.temp + 2.5f;
  s.v_bat    = 11.4f;
  s.i_bat    = 0.02f;
  s.v_uc     = 5.1f;
  s.i_uc     = 0.0f;
  s.v_pv     = (s.irr > 50.0f) ? 17.5f : 1.0f;
  s.i_pv     = (s.irr > 50.0f) ? 0.8f : 0.0f;
#else
  s.temp     = bme.readTemperature();
  s.hum      = bme.readHumidity();
  s.pres     = bme.readPressure() / 100.0f;
  s.irr      = 0.0f; // Add BH1750 or LDR read here
  s.enc_temp = s.temp;
  s.v_bat    = ina_bat.getBusVoltage_V();
  s.i_bat    = ina_bat.getCurrent_mA() / 1000.0f;
  s.v_uc     = ina_uc.getBusVoltage_V();
  s.i_uc     = ina_uc.getCurrent_mA() / 1000.0f;
  s.v_pv     = ina_pv.getBusVoltage_V();
  s.i_pv     = ina_pv.getCurrent_mA() / 1000.0f;
#endif
  s.p_pv    = fmaxf(0.0f, s.v_pv * s.i_pv);
  s.bat_soc = constrain((s.v_bat - 9.0f) / (12.6f - 9.0f), 0.0f, 1.0f);
  return s;
}

// ==================== FEATURES, CHANGE GATE & TARGETS ====================
void buildDynamicFeatures(const SensorSnapshot& s, float x_dyn[6]) {
  float vol = fmaxf(fabsf(s.temp - last_tx_temp) / 10.0f,
              fmaxf(fabsf(s.hum  - last_tx_hum)  / 30.0f,
                    fabsf(s.pres - last_tx_pres) / 10.0f));
  float hour = fmodf((boot_count * 5.0f) / 60.0f, 24.0f);

  x_dyn[0] = s.bat_soc;
  x_dyn[1] = constrain(s.v_uc / V_UC_MAX, 0.0f, 1.0f);
  x_dyn[2] = constrain(s.p_pv / 40.0f, 0.0f, 1.0f);
  x_dyn[3] = constrain(vol, 0.0f, 1.0f);
  x_dyn[4] = 0.5f * (1.0f + sinf(2.0f * 3.14159f * hour / 24.0f));
  x_dyn[5] = 0.5f * (1.0f + cosf(2.0f * 3.14159f * hour / 24.0f));
}

bool checkChangeGate(const SensorSnapshot& s) {
  float max_dev = fmaxf(fabsf(s.temp - last_tx_temp) / 5.0f,
                  fmaxf(fabsf(s.hum  - last_tx_hum)  / 20.0f,
                  fmaxf(fabsf(s.pres - last_tx_pres) / 5.0f,
                        fabsf(s.irr  - last_tx_irr)  / 300.0f)));
  return (max_dev >= active_delta) || (mins_since_tx >= (uint32_t)active_dt_tx_min);
}

void computeOptimalTargets(const SensorSnapshot& s, const float x_dyn[6], float y_target[3]) {
  float energy_score = 0.6f * s.bat_soc + 0.4f * x_dyn[2];
  y_target[0] = constrain(1.0f - energy_score, 0.05f, 0.95f);
  y_target[1] = constrain(0.75f - 0.60f * energy_score, 0.10f, 0.85f);
  float uc_low = (s.v_uc < 0.85f * V_UC_MAX) ? 1.0f : 0.0f;
  y_target[2] = (uc_low > 0.5f && s.bat_soc > 0.35f) ? 1.0f : 0.0f;
}

// ==================== WI-FI & FASTAPI CLOUD SYNC ====================
bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000) {
    delay(200);
  }
  return WiFi.status() == WL_CONNECTED;
}

bool beginHttp(HTTPClient& http, WiFiClientSecure& tlsClient, const String& endpoint) {
  String fullUrl = String(SERVER_URL) + endpoint;
  if (fullUrl.startsWith("https://")) {
    tlsClient.setInsecure(); // Swap with setCACert(root_ca) for production TLS pinning
    return http.begin(tlsClient, fullUrl);
  }
  return http.begin(fullUrl);
}

void sendWeatherTelemetry(const SensorSnapshot& s, bool used_uc) {
  WiFiClientSecure tls;
  HTTPClient http;
  if (!beginHttp(http, tls, "/telemetry")) return;
  http.addHeader("Content-Type", "application/json");

  JsonDocument doc;
  doc["node_id"]       = NODE_ID;
  doc["temperature"]   = s.temp;
  doc["humidity"]      = s.hum;
  doc["pressure"]      = s.pres;
  doc["irradiance"]    = s.irr;
  doc["v_bat"]         = s.v_bat;
  doc["v_uc"]          = s.v_uc;
  doc["v_pv"]          = s.v_pv;
  doc["used_ultracap"] = used_uc ? 1 : 0;

  String body;
  serializeJson(doc, body);
  int code = http.POST(body);
  Serial.printf("[TX] /telemetry -> HTTP %d (%d bytes)\n", code, body.length());
  http.end();
}

float trainOnLocalLogs() {
  int count = getLoggedSampleCount();
  if (count < MIN_SAMPLES_FL) return -1.0f;

  File f = LittleFS.open("/logs.bin", "r");
  if (!f) return -1.0f;

  uint32_t t_start = millis();
  float last_loss = 0.0f;
  SampleRecord rec;

  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    f.seek(0);
    float epoch_loss = 0.0f;
    int n = 0;
    while (f.read((uint8_t*)&rec, sizeof(SampleRecord)) == sizeof(SampleRecord)) {
      epoch_loss += mlp.trainStep(rec.x_dyn, STAT_META, rec.y_target, LEARNING_RATE, FEDPROX_MU);
      n++;
    }
    last_loss = (n > 0) ? (epoch_loss / n) : 0.0f;
  }
  f.close();
  Serial.printf("[FL] Trained %d epochs on %d samples in %lu ms | Loss: %.5f\n",
                LOCAL_EPOCHS, count, millis() - t_start, last_loss);
  return last_loss;
}

void syncFederatedModel() {
  WiFiClientSecure tls;
  HTTPClient http;

  // 1. Check server version
  int server_round = -1;
  if (beginHttp(http, tls, "/fl/version")) {
    if (http.GET() == 200) {
      JsonDocument doc;
      if (!deserializeJson(doc, http.getString())) {
        server_round = doc["round"] | -1;
      }
    }
    http.end();
  }

  // 2. Download global model if server is ahead
  if (server_round > current_round && beginHttp(http, tls, "/fl/model")) {
    if (http.GET() == 200) {
      JsonDocument doc;
      if (!deserializeJson(doc, http.getString())) {
        JsonArray arr = doc["weights"].as<JsonArray>();
        const char* srv_hash = doc["sha256"] | "";
        if (arr.size() == NUM_WEIGHTS) {
          float cand[NUM_WEIGHTS];
          bool finite_ok = true;
          for (int i = 0; i < NUM_WEIGHTS; i++) {
            cand[i] = arr[i].as<float>();
            if (!isfinite(cand[i])) finite_ok = false;
          }
          char calc_hash[65];
          computeSha256Hex(cand, NUM_WEIGHTS, calc_hash);
          if (finite_ok && strcasecmp(calc_hash, srv_hash) == 0) {
            memcpy(mlp.w, cand, sizeof(cand));
            memcpy(mlp.w_global, cand, sizeof(cand));
            current_round = doc["round"] | server_round;
            saveWeightsToFlash();
            Serial.printf("[FL] Downloaded verified global round %d\n", current_round);
          }
        }
      }
    }
    http.end();
  }

  // 3. Train locally & upload weights
  int n_samples = getLoggedSampleCount();
  float local_loss = trainOnLocalLogs();
  if (local_loss >= 0.0f && isfinite(local_loss)) {
    char hash_hex[65];
    computeSha256Hex(mlp.w, NUM_WEIGHTS, hash_hex);

    JsonDocument doc;
    doc["node_id"]     = NODE_ID;
    doc["round"]       = max(current_round, 0);
    doc["num_samples"] = n_samples;
    doc["local_loss"]  = local_loss;
    doc["sha256"]      = hash_hex;
    JsonArray arr = doc["weights"].to<JsonArray>();
    for (int i = 0; i < NUM_WEIGHTS; i++) arr.add(mlp.w[i]);

    String payload;
    serializeJson(doc, payload);

    if (beginHttp(http, tls, "/fl/update")) {
      http.addHeader("Content-Type", "application/json");
      int code = http.POST(payload);
      Serial.printf("[FL] /fl/update -> HTTP %d (%d bytes) | %s\n",
                    code, payload.length(), http.getString().c_str());
      http.end();
    }
    saveWeightsToFlash();
  }
}

// ==================== MAIN WAKE-UP CYCLE ====================
void setup() {
  Serial.begin(115200);
  delay(200);
  initSwitchPins();
  Wire.begin(21, 22);
  LittleFS.begin(true);

#if !USE_MOCK_SENSORS
  ina_bat.begin();
  ina_uc.begin();
  ina_pv.begin();
  bme.begin(0x76);
#endif

  if (!loadWeightsFromFlash()) {
    mlp.initRandom();
    saveWeightsToFlash();
  }

  boot_count++;
  mins_since_tx += 5;

  SensorSnapshot snap = readAllSensors();
  float x_dyn[6];
  buildDynamicFeatures(snap, x_dyn);

  uint32_t t0_us = micros();
  mlp.forward(x_dyn, STAT_META);
  uint32_t infer_us = micros() - t0_us;

  active_dt_tx_min = 15.0f + mlp.out[0] * (120.0f - 15.0f);
  active_delta     = 0.05f + mlp.out[1] * 0.95f;
  active_pi_chg    = mlp.out[2];

  SampleRecord rec;
  memcpy(rec.x_dyn, x_dyn, sizeof(x_dyn));
  computeOptimalTargets(snap, x_dyn, rec.y_target);
  appendLogSample(rec);

  manageSolarCharging(snap, active_pi_chg);

  Serial.printf("\n=== Boot #%lu | V_bat=%.2fV V_uc=%.2fV P_pv=%.2fW ===\n",
                boot_count, snap.v_bat, snap.v_uc, snap.p_pv);
  Serial.printf("[MLP] Inference: %lu us | dT_tx=%.1f min, delta=%.2f, pi_chg=%.2f\n",
                infer_us, active_dt_tx_min, active_delta, active_pi_chg);
  Serial.printf("[MEM] Free Heap: %lu bytes | LittleFS Used: %u bytes | Samples: %d\n",
                ESP.getFreeHeap(), (unsigned)LittleFS.usedBytes(), getLoggedSampleCount());

  bool do_tx      = checkChangeGate(snap);
  bool do_fl_sync = (snap.p_pv > 5.0f || BENCH_TEST_MODE)
                    && (snap.v_bat > 10.0f)
                    && ((boot_count - last_fl_boot) >= FL_SYNC_EVERY);

  if (do_tx || do_fl_sync) {
    if (snap.v_bat > V_BAT_MIN || snap.v_uc >= V_UC_MIN) {
      bool on_uc = switchToUltracapTX(snap.v_uc);
      Serial.printf("[PWR] Radio TX using: %s\n", on_uc ? "1500F ULTRACAP" : "3S2P BATTERY FALLBACK");

      if (connectWiFi()) {
        if (do_tx) {
          sendWeatherTelemetry(snap, on_uc);
          last_tx_temp  = snap.temp;
          last_tx_hum   = snap.hum;
          last_tx_pres  = snap.pres;
          last_tx_irr   = snap.irr;
          mins_since_tx = 0;
        }
        if (do_fl_sync) {
          syncFederatedModel();
          last_fl_boot = boot_count;
        }
        WiFi.disconnect(true);
      } else {
        Serial.println("[WIFI] Connection failed; retaining local model.");
      }

      if (on_uc) switchToBatteryBase();
    }
  } else {
    Serial.println("[GATE] No significant weather change; TX suppressed.");
  }

  // Hold active charging switch state across deep sleep
  gpio_hold_en((gpio_num_t)PIN_SW_CHG_BAT);
  gpio_hold_en((gpio_num_t)PIN_SW_CHG_UC);
  gpio_deep_sleep_hold_en();

  WiFi.mode(WIFI_OFF);
  Serial.printf("Sleeping for %lu s...\n", WAKE_INTERVAL_SEC);
  Serial.flush();

  esp_sleep_enable_timer_wakeup((uint64_t)WAKE_INTERVAL_SEC * 1000000ULL);
  esp_deep_sleep_start();
}

void loop() {
  // Deep sleep wakes directly into setup()
}