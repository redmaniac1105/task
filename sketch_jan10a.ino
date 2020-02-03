void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
pinMode(0,OUTPUT);
pinMode(1,OUTPUT);
pinMode(2,OUTPUT);
pinMode(3,OUTPUT);
pinMode(4,OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int i,num,li[5];
  num=Serial.parseInt();
  for(i=5;i>0;i--)
  {
    li[i]=num%2;
    num=num/2;
  }
  for(i=0;i<5;i++)
  {
    if(li[i]==0)
      digitalWrite(i,LOW);
    else
      digitalWrite(i,HIGH);
  }

}
