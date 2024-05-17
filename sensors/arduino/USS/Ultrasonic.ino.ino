const int trigPin = 9;
const int echoPin = 10;
float duration;
int distance;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
 
   
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  int original_thickness = 5;
  int total_distance = 20;

  duration = pulseIn(echoPin, HIGH);
  //Serial.print("Sound travel time:");
  // Serial.println(duration);

  distance = duration * 0.034/ 2;

  if ( abs(distance - total_distance) > original_thickness) {
     Serial.println("9"); //9 means hole detected
    // dist_tot = og_dist + og_thickness
  } else {
    int current_thickness = total_distance - distance;
    //Serial.print("Current thickness is: ");
    // Serial.println(current_thickness);
    int tension = map(current_thickness, 0, 5, 7, 0); 
    //Serial.print("Tension: ");
    // Serial.println(tension);
    String duration_1 = String(duration);
    String current_thickness_1 = String(current_thickness);
    String tension_1 = String(tension);
    String abcd = duration_1 + " " + current_thickness_1 + " " + tension_1;
    Serial.println(abcd);
  }
  delay(1000);
}
