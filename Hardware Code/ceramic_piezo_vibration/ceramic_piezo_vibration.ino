void setup() {
  Serial.begin(115200);
  Serial.print("Detact Vibration Start!!!");
}

void loop() {
  int val = analogRead(35);
  Serial.println(val);
  if (val > 100) {
    Serial.println("Detect");
  }
  delay(100);
}
