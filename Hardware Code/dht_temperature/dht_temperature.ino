#include "DHT.h"

#define DHTPIN 5       // Pin connected to the DHT sensor (ESP32 pin D5)
#define DHTTYPE DHT11  // Specify the type of DHT sensor: DHT22, DHT11, etc.

// Initialize the DHT sensor
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  // Start the Serial Monitor
  Serial.begin(115200);
  Serial.println(F("DHT Sensor Test with ESP32"));

  // Initialize the DHT sensor
  dht.begin();
}

void loop() {
  // Wait a few seconds between measurements
  delay(2000);

  // Reading temperature and humidity from the sensor
  float humidity = dht.readHumidity();
  float temperatureC = dht.readTemperature();    // Temperature in Celsius
  float temperatureF = dht.readTemperature(true); // Temperature in Fahrenheit

  // Check if the readings failed
  if (isnan(humidity) || isnan(temperatureC) || isnan(temperatureF)) {
    Serial.println(F("Failed to read from DHT sensor! Check wiring."));
    return;
  }

  // Compute the heat index
  float heatIndexC = dht.computeHeatIndex(temperatureC, humidity, false); // Heat index in Celsius
  float heatIndexF = dht.computeHeatIndex(temperatureF, humidity);       // Heat index in Fahrenheit

  // Print the readings to the Serial Monitor
  Serial.print(F("Humidity: "));
  Serial.print(humidity);
  Serial.print(F("%  Temperature: "));
  Serial.print(temperatureC);
  Serial.print(F("°C "));
  Serial.print(temperatureF);
  Serial.print(F("°F  Heat Index: "));
  Serial.print(heatIndexC);
  Serial.print(F("°C "));
  Serial.print(heatIndexF);
  Serial.println(F("°F"));
}
