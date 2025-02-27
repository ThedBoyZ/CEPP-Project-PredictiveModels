void setup() {
    Serial.begin(115200);
}

void loop() {
    int analogValue = analogRead(34); 
    Serial.println(analogValue); 
    delay(10); 
}
