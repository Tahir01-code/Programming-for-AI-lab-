from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Hotel chatbot responses
hotel_responses = {
    "greeting": ["Hello! Welcome to Grand Orchard Hotel! 🏨 How can I assist you today?",
                 "Good day! I'm your virtual concierge at Grand Orchard Hotel. How may I help you?"],

    "rooms": """🛏️ **Our Room Types:**

• **Standard Room** – PKR 11,000/night
  - 1 Queen Bed, AC, WiFi, TV
  
• **Deluxe Room** – PKR 16,000/night
  - 1 King Bed, City View, Minibar, AC, WiFi
  
• **Junior Suite** – PKR 22,000/night
  - Living Area, King Bed, Jacuzzi, Room Service
  
• **Presidential Suite** – PKR 56,000/night
  - 2 Bedrooms, Private Lounge, Butler Service, Panoramic View

Would you like to make a booking?""",

    "booking": """📅 **To Book a Room:**

1. Call us: **+92-300-1234567**
2. Email: **tahirofficial458@gmail.com**
3. Visit our website: **www.grandorchardhotel.pk**

**Check-in:** 3:00 PM | **Check-out:** 11:00 AM

Need help choosing a room?""",

    "amenities": """✨ **Hotel Amenities:**

🏊 Swimming Pool (6 AM – 10 PM)
🏋️ Fitness Center (24/7)
🍽️ Restaurant & Rooftop Café
💆 Spa & Wellness Center
🅿️ Free Parking
📶 Free High-Speed WiFi
🚗 Airport Shuttle Service
👶 Kids Play Area
💼 Business Center & Conference Rooms

Which amenity would you like to know more about?""",

    "restaurant": """🍽️ **Dining at Grand Orchard Hotel:**

**Spice Garden Restaurant**
- Breakfast: 7 AM – 10:30 AM
- Lunch: 12 PM – 3 PM  
- Dinner: 7 PM – 11 PM
- Cuisine: Pakistani, Chinese, Continental

**Rooftop Café**
- Open: 4 PM – 12 AM
- Light bites, beverages & BBQ

**Room Service:** Available 24/7 🕐

Want to make a restaurant reservation?""",

    "price": """💰 **Room Prices:**

| Room Type      | Per Night     |
|---------------|---------------|
| Standard      | PKR 11,000    |
| Deluxe        | PKR 16,000    |
| Junior Suite  | PKR 22,000    |
| Presidential  | PKR 56,000    |

*Prices include breakfast & WiFi*
*10% discount on 3+ nights stay!*

Want to make a booking?""",

    "location": """📍 **Hotel Location:**

Grand Orchard Hotel
Main Boulevard, Gulberg III
Lahore, Pakistan 54000

🚗 **Nearby:**
- 25 min from Allama Iqbal Airport
- 10 min from Liberty Market
- 5 min from Packages Mall

**GPS:** 31.5204° N, 74.3587° E

Need directions or airport pickup?""",

    "wifi": """📶 **WiFi Information:**

Free high-speed WiFi throughout the hotel!

**Network:** GrandOrchardHotel_Guest
**Password:** Given at check-in

Speed: Up to 100 Mbps
Coverage: All rooms, lobby, pool area, restaurant

Is there anything else I can help you with?""",

    "spa": """💆 **Spa & Wellness Center:**

**Services:**
- Full Body Massage – PKR 4,500
- Facial Treatment – PKR 3,500
- Couple's Spa Package – PKR 7,000
- Aromatherapy – PKR 3,800

**Hours:** 9 AM – 9 PM (Daily)

📞 Book via reception or call ext. 205

Want to schedule a session?""",

    # BUG FIX: Check-out time was "11:00 PM" — corrected to "11:00 AM"
    "checkout": """🏁 **Check-out Information:**

- **Check-out Time:** 11:00 AM
- **Late Check-out:** PKR 5,000 extra (subject to availability)
- **Express Check-out:** Available 24/7

**On Departure:**
✅ Return room keys at reception
✅ Settle any remaining bills
✅ Request airport transfer if needed

Need anything else before your stay ends?""",

    "pool": """🏊 **Swimming Pool:**

**Hours:** 7:00 AM – 10:00 PM
**Location:** 3rd Floor (Outdoor)
**Temperature:** Heated (27°C year-round)

Includes:
- Towels provided free
- Poolside bar & snacks
- Lifeguard on duty
- Kids' splash area

Enjoy your swim! 🌊""",

    "default": ["I'm not sure I understood that. You can ask me about:\n\n🛏️ Rooms & Prices\n📅 Booking\n✨ Amenities\n🍽️ Restaurant\n📍 Location\n💆 Spa\n🏊 Pool\n📶 WiFi\n🏁 Check-out",
                "Sorry, I didn't catch that! Try asking about rooms, booking, amenities, restaurant, location, spa, or wifi. 😊"]
}


def get_response(user_message):
    msg = user_message.lower().strip()

    if any(word in msg for word in ["hi", "hello", "hey", "salam", "assalam", "good morning", "good evening"]):
        return random.choice(hotel_responses["greeting"])
    elif any(word in msg for word in ["room", "rooms", "type", "stay", "bed", "suite"]):
        return hotel_responses["rooms"]
    elif any(word in msg for word in ["book", "booking", "reserve", "reservation", "availability", "available"]):
        return hotel_responses["booking"]
    elif any(word in msg for word in ["ameniti", "facility", "facilities", "services", "feature"]):
        return hotel_responses["amenities"]
    elif any(word in msg for word in ["restaurant", "food", "eat", "dining", "breakfast", "lunch", "dinner", "cafe", "menu"]):
        return hotel_responses["restaurant"]
    elif any(word in msg for word in ["price", "cost", "rate", "charge", "fee", "tariff", "how much"]):
        return hotel_responses["price"]
    elif any(word in msg for word in ["location", "address", "where", "direction", "map", "place", "situated"]):
        return hotel_responses["location"]
    elif any(word in msg for word in ["wifi", "internet", "network", "wi-fi", "wireless", "password"]):
        return hotel_responses["wifi"]
    elif any(word in msg for word in ["spa", "massage", "facial", "wellness", "relax", "treatment"]):
        return hotel_responses["spa"]
    elif any(word in msg for word in ["checkout", "check out", "check-out", "leave", "departure", "late"]):
        return hotel_responses["checkout"]
    elif any(word in msg for word in ["pool", "swim", "swimming", "jacuzzi"]):
        return hotel_responses["pool"]
    elif any(word in msg for word in ["thanks", "thank you", "shukriya", "shukria", "bye", "goodbye", "ok", "okay"]):
        return "You're welcome! 😊 It was a pleasure assisting you. Enjoy your stay at Grand Orchard Hotel! 🏨✨"
    else:
        return random.choice(hotel_responses["default"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    response = get_response(user_message)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)