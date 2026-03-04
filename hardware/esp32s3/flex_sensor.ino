#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid     = "your_wifi";
const char* password = "your_password";
const char* hostIP   = "your_ip";
const int   udpPort  = 1400;

WiFiUDP udp;
const int flexPin = 1; //A0
const int ledPin  = 21; //LED

int loopCounter = 0;
bool ledState = false;

void setup() {
  Serial.begin(115200);

  pinMode(ledPin, OUTPUT);
  pinMode(flexPin, INPUT);

  digitalWrite(ledPin, HIGH); //High=Disable

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.print("Connecting to WiFi");
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(ledPin, LOW);
    delay(100);
    digitalWrite(ledPin, HIGH);
    delay(100);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! XIAO IP: ");
  Serial.println(WiFi.localIP());
  digitalWrite(ledPin, LOW);
  delay(1000);

  udp.begin(udpPort);
}

void loop() {
  int rawValue = analogRead(flexPin);
  float voltage = rawValue * (3.3 / 4095.0);

  udp.beginPacket(hostIP, udpPort);
  udp.printf("FLEX:%d", rawValue);
  udp.endPacket();

  loopCounter++;
  if (loopCounter >= 25) {
    ledState = !ledState;
    digitalWrite(ledPin, ledState ? LOW : HIGH);
    loopCounter = 0;
  }

  delay(20);
}