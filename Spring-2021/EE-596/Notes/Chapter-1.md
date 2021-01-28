# Chapter 1 - Radar Basics

## 1.1 Introduction
Radar is an acronym for Radio Detection and Ranging. Radar has been around since 1904, when Christian Hulsmeyer bounced waves off a a ship. Later during the 1920s, interest in this technology started to increase, and really started taking off during WWII, resulting in the widespread adoption we see today. 

## 1.2 Radar Types
* Radar can use two types of signals:
  * Pulsed signals - transmit a sequence of radio frequency (RF) energy. An antenna connects to the transmitter as it sends signals out that connect to the reciever after a transmission phase. Transmit/recieve (T/R) switches handle this switching functions. These are the most common types of signals. 

  * Continuous Waves (CW) - transmits a continous signal, typically using two antennas. One for recieving and one for transmission.

* Types of Radar
  * Monostatic - The transmitter and the reciever and placed together, making them compact. Which makes them the most common type.

  * Bistatic - The transmitter and the reciever are physically seperated, often over long distances. These have more specific applications such as in aircraft or missles. 

While the overall goal of radar is to detect targets and thier location, they can also determine the distance between the two as well as the angle. This results in a location and velocity values, and sometimes higher derivatives of the location. 

Radar operates in the radio band of the electromagnetic spectrum between 5 MHz and 300 GHz. Often times search radar will operate in C band, tracking radar in S-, Ku-, K-, and Ka bands, and intrumentation and short-range radar in Ka or above. The table below shows the bands as identified by IEEE. 

|Band | Frequency Range |
|-----|-----------------|
| HF  | 3-30 Mhz        |
| VHF | 30-300 Mhz      |
| UHF | 300-1000 Mhz    |
| L   | 1-2 Ghz         |
| S   | 2-4 Ghz         |
| C   | 4-8 Ghz         |
| X   | 8-12 Ghz        |
| Ku  | 12-18 Ghz       |
| K   | 18-27 Ghz       |
| Ka  | 27-40 Ghz       |
| V   | 40-75 Ghz       |
| W   | 75-110 Ghz      |
| mm  | 110-300 Ghz     |

## 1.3 Range Measurement
The most common way to measure range with radar is to measure the difference in time from transmission to reception of a pulse. Radio Frequencies (RF) travels at the speed of light (3x10^8 m/s).

```
  R = Range of the target
  c = speed of light 

  time to target (t1) = R / c
  time from target (t2) = R / c

  total time (t3) = t1 + t2
  
  R = c * t3 / 2

```
It is worth noting that the expression of time delay can be expressed in many different scales, however microseconds is common. Consider how the range measurment can be derived, with a time scale in microseconds. 

```
  R = (c / 2) * total time 

  Example:
  (( 3 x 10^8) / 2) *  (total time * 10^-6) = 150

```
In other words to convert the total time to microseconds, scale total time by 150. 

## 1.4 Ambiguous Range
Due to the nature of pulse-based radar, determining the range to a target becomes difficult. A common practice is to set t (the time of the transmission pulse) to 0 and to reset it to zero after every pulse. Assuming we have pulses spaced 400 microseconds apart, at a target range of 90km. 

```
  Target Range Delay  = 2R / c 
                      = 2x90x^3/3x10^8 
                      = 60x10^-5 
                      = 600 microseconds
```

In this context the return from the first pulse is not recieved until after a second pulse is already sent out. Since they are coming and going at different times, there is no way to determine which pulse corresponds to each response. Because the timer is being reset with each pulse transmissin, it would record the target range incorrectly. This is ambigious, and makes a high level of uncertainty when measuring range. We can calculate this by using the following formula. 

```
  Unambiguous Range = (c * Spacing between pulses) / 2

```

If the target range is less than this value, the range can be measured unambiguously. If the target range is greather than this value, the range cannot measure range unambiguously. The spacing between transmission intervals is reffered to as pulse repetition interval. To avoid range ambigiuities, radar engineers typically use PRI to excee

## 1.5 Usable Range and Instrumented Range

## 1.6 Range-Rate Measurement (Doppler)