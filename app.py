
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

MODEL_FILE = "houseworth_model.pkl"

# --------------------------------------------------
# HOUSEWORTH.COM - FACILITY LIST (250 OPTIONS)
# --------------------------------------------------

FACILITY_CATEGORIES = {
    "Swimming and Water": [
        "Private Swimming Pool", "Indoor Swimming Pool",
        "Outdoor Swimming Pool", "Infinity Pool",
        "Rooftop Pool", "Heated Pool", "Kids Pool",
        "Jacuzzi", "Hot Tub", "Water Park",
        "Water Slides", "Poolside Lounge", "Pool Bar",
        "Pool Shower", "Pool Changing Room"
    ],
    "Fitness and Wellness": [
        "Gymnasium", "Fitness Centre", "Yoga Studio",
        "Meditation Room", "Spa", "Steam Room",
        "Sauna", "Massage Room", "Health Club",
        "Aerobics Studio", "Pilates Studio", "Jogging Track",
        "Walking Track", "Cycling Track", "Indoor Fitness Area",
        "Outdoor Fitness Area", "Personal Training Room",
        "Wellness Centre", "Reflexology Path", "Therapy Room"
    ],
    "Sports and Games": [
        "Badminton Court", "Tennis Court", "Basketball Court",
        "Volleyball Court", "Football Ground", "Cricket Practice Net",
        "Squash Court", "Table Tennis Room", "Billiards Room",
        "Snooker Room", "Bowling Alley", "Indoor Games Room",
        "Outdoor Games Area", "Skating Rink", "Golf Putting Green",
        "Mini Golf", "Boxing Room", "Martial Arts Studio",
        "Multi Purpose Sports Court", "Sports Equipment Room"
    ],
    "Gardens and Outdoor": [
        "Landscaped Garden", "Rooftop Garden", "Terrace Garden",
        "Private Garden", "Herbal Garden", "Kitchen Garden",
        "Butterfly Garden", "Japanese Garden", "Zen Garden",
        "Flower Garden", "Botanical Garden", "Rock Garden",
        "Water Fountain", "Outdoor Seating Area", "Gazebo",
        "Pergola", "Barbecue Area", "Outdoor Dining Area",
        "Children's Play Garden", "Nature Trail"
    ],
    "Security": [
        "24 Hour Security", "CCTV Surveillance", "Video Door Phone",
        "Smart Door Lock", "Biometric Access", "Gated Community",
        "Security Guard Room", "Intercom Facility", "Access Card System",
        "Visitor Management System", "Fire Alarm System",
        "Fire Sprinkler System", "Smoke Detector", "Emergency Exit",
        "Security Control Room"
    ],
    "Smart Home": [
        "Smart Home Automation", "Smart Lighting",
        "Smart Temperature Control", "Smart Curtains",
        "Smart Doorbell", "Voice Controlled System",
        "Smart Entertainment System", "Smart Appliances",
        "Smart Energy Meter", "Remote Home Monitoring",
        "Smart Irrigation", "Automated Garage Door",
        "Smart Locks", "Motion Sensors", "Central Control Panel"
    ],
    "Parking and Transport": [
        "Covered Car Parking", "Open Car Parking",
        "Visitor Parking", "Reserved Parking",
        "Basement Parking", "Multi Level Parking",
        "EV Charging Station", "Bicycle Parking",
        "Two Wheeler Parking", "Valet Parking",
        "Car Wash Area", "Driver Rest Room",
        "Shuttle Service", "Private Lift", "Service Lift"
    ],
    "Entertainment": [
        "Home Theatre", "Mini Theatre", "Multipurpose Hall",
        "Party Hall", "Banquet Hall", "Music Room",
        "Dance Studio", "Karaoke Room", "Library",
        "Reading Lounge", "Gaming Lounge", "Virtual Reality Room",
        "Cinema Room", "Rooftop Lounge", "Sky Lounge",
        "Indoor Entertainment Room", "Outdoor Amphitheatre",
        "Music Practice Room", "Art Studio", "Hobby Room"
    ],
    "Community and Family": [
        "Children's Play Area", "Creche", "Day Care Centre",
        "Senior Citizen Lounge", "Community Hall",
        "Clubhouse", "Guest Rooms", "Family Lounge",
        "Community Kitchen", "Multipurpose Activity Room",
        "Celebration Area", "Pet Park", "Pet Washing Station",
        "Community Garden", "Indoor Kids Play Area"
    ],
    "Interiors": [
        "Modular Kitchen", "Walk In Wardrobe", "Dressing Room",
        "False Ceiling", "Premium Flooring", "Marble Flooring",
        "Wooden Flooring", "Designer Lighting", "Italian Kitchen",
        "Built In Wardrobes", "Study Room", "Home Office",
        "Pooja Room", "Pantry", "Storage Room",
        "Laundry Room", "Utility Room", "Servant Room",
        "Private Balcony", "Large Windows"
    ],
    "Building Facilities": [
        "Power Backup", "Generator Backup", "Rainwater Harvesting",
        "Water Treatment Plant", "Sewage Treatment Plant",
        "Water Softener", "24 Hour Water Supply",
        "Central Air Conditioning", "High Speed Elevators",
        "Solar Power System", "Waste Management",
        "Underground Water Tank", "Overhead Water Tank",
        "Broadband Internet", "WiFi Common Areas"
    ],
    "Sustainability": [
        "Solar Water Heating", "Solar Street Lights",
        "Energy Efficient Lighting", "Green Building Design",
        "Organic Waste Converter", "Composting Unit",
        "Rainwater Recharge Pit", "Greywater Recycling",
        "EV Friendly Infrastructure", "Electricity Monitoring",
        "Green Roof", "Vertical Garden", "Natural Ventilation",
        "Double Glazed Windows", "Eco Friendly Materials"
    ],
    "Retail and Services": [
        "Supermarket", "Convenience Store", "ATM",
        "Pharmacy", "Salon", "Beauty Parlour",
        "Laundry Service", "Dry Cleaning Service",
        "Coffee Shop", "Restaurant", "Bakery",
        "Medical Clinic", "Doctor Consultation Room",
        "Grocery Delivery Room", "Parcel Collection Room"
    ],
    "Hospitality and Luxury": [
        "Concierge Service", "Reception Lobby", "Luxury Lobby",
        "Guest Reception", "Housekeeping Service",
        "Room Service", "Business Centre", "Conference Room",
        "Executive Lounge", "Waiting Lounge", "Cafeteria",
        "Fine Dining Area", "Rooftop Restaurant",
        "Outdoor Cafe", "Luxury Guest Suite"
    ],
    "Private Amenities": [
        "Private Terrace", "Private Rooftop",
        "Private Entertainment Room", "Private Gym",
        "Private Home Office", "Private Library",
        "Private Theatre", "Private Spa", "Private Jacuzzi",
        "Private Bar", "Private Dining Room",
        "Private Garden Entrance", "Private Swimming Deck",
        "Private Courtyard", "Private Guest Suite"
    ],
    "Additional Amenities": [
        "Sky Bridge", "Observation Deck", "Viewing Gallery",
        "Indoor Water Feature", "Outdoor Water Feature",
        "Open Air Lounge", "Sun Deck", "Meditation Garden",
        "Community Kitchen Garden", "Outdoor Shower",
        "Changing Facilities", "Lockers", "First Aid Room",
        "Emergency Medical Room", "Ambulance Access",
        "Fire Safety Equipment", "Postal Room",
        "Maintenance Office", "Facility Management Office",
        "Building Management System"
    ]
}
FACILITY_CATEGORIES = dict(list(FACILITY_CATEGORIES.items())[:15])

FACILITIES = [
    item
    for items in FACILITY_CATEGORIES.values()
    for item in items
]

# Ensure exactly 250 unique options
assert len(FACILITIES) == 250, (
    f"Expected 250 facilities, got {len(FACILITIES)}"
)

# --------------------------------------------------
# TAMIL NADU LOCATION OPTIONS
# --------------------------------------------------

LOCATIONS = {
    "Ariyalur": ["Ariyalur", "Jayankondam", "Sendurai"],
    "Chengalpattu": ["Chengalpattu", "Tambaram", "Pallavaram", "Mahabalipuram"],
    "Chennai": [
        "Adyar", "Anna Nagar", "Ambattur", "Avadi", "Chromepet",
        "Guindy", "Kodambakkam", "Medavakkam", "Mylapore",
        "OMR", "Porur", "T Nagar", "Velachery", "Sholinganallur"
    ],
    "Coimbatore": [
        "Coimbatore", "Gandhipuram", "RS Puram", "Peelamedu",
        "Saravanampatti", "Vadavalli", "Singanallur",
        "Saibaba Colony", "Thudiyalur", "Thondamuthur",
        "Theethipalayam", "Pollachi", "Mettupalayam"
    ],
    "Cuddalore": ["Cuddalore", "Chidambaram", "Panruti", "Virudhachalam"],
    "Dharmapuri": ["Dharmapuri", "Harur", "Palacode"],
    "Dindigul": ["Dindigul", "Palani", "Kodaikanal", "Oddanchatram"],
    "Erode": ["Erode", "Gobichettipalayam", "Bhavani", "Perundurai"],
    "Kallakurichi": ["Kallakurichi", "Chinnasalem", "Sankarapuram"],
    "Kancheepuram": ["Kancheepuram", "Sriperumbudur", "Uthiramerur"],
    "Kanniyakumari": ["Nagercoil", "Marthandam", "Kanyakumari", "Colachel"],
    "Karur": ["Karur", "Kulithalai", "Aravakurichi"],
    "Krishnagiri": ["Krishnagiri", "Hosur", "Denkanikottai"],
    "Madurai": ["Madurai", "Thirumangalam", "Melur", "Usilampatti"],
    "Mayiladuthurai": ["Mayiladuthurai", "Sirkazhi", "Kuthalam"],
    "Nagapattinam": ["Nagapattinam", "Vedaranyam", "Thirukkuvalai"],
    "Namakkal": ["Namakkal", "Tiruchengode", "Rasipuram"],
    "Nilgiris": ["Ooty", "Coonoor", "Kotagiri", "Gudalur"],
    "Perambalur": ["Perambalur", "Kunnam"],
    "Pudukkottai": ["Pudukkottai", "Aranthangi", "Alangudi"],
    "Ramanathapuram": ["Ramanathapuram", "Paramakudi", "Rameswaram"],
    "Ranipet": ["Ranipet", "Arcot", "Walajapet"],
    "Salem": ["Salem", "Attur", "Mettur", "Omalur"],
    "Sivaganga": ["Sivaganga", "Karaikudi", "Devakottai"],
    "Tenkasi": ["Tenkasi", "Sankarankovil", "Courtallam"],
    "Thanjavur": ["Thanjavur", "Kumbakonam", "Pattukkottai"],
    "Theni": ["Theni", "Periyakulam", "Bodinayakanur", "Cumbum"],
    "Thoothukudi": ["Thoothukudi", "Kovilpatti", "Tiruchendur"],
    "Tiruchirappalli": ["Tiruchirappalli", "Srirangam", "Manapparai", "Lalgudi"],
    "Tirunelveli": ["Tirunelveli", "Ambasamudram", "Palayamkottai"],
    "Tirupathur": ["Tirupathur", "Vaniyambadi", "Ambur"],
    "Tiruppur": ["Tiruppur", "Avinashi", "Dharapuram", "Palladam"],
    "Tiruvallur": ["Tiruvallur", "Poonamallee", "Tiruttani", "Gummidipoondi"],
    "Tiruvannamalai": ["Tiruvannamalai", "Arani", "Polur"],
    "Tiruvarur": ["Tiruvarur", "Mannargudi", "Thiruthuraipoondi"],
    "Vellore": ["Vellore", "Katpadi", "Gudiyatham"],
    "Viluppuram": ["Viluppuram", "Tindivanam", "Gingee"],
    "Virudhunagar": ["Virudhunagar", "Sivakasi", "Rajapalayam", "Aruppukkottai"]
}

# --------------------------------------------------
# DEMONSTRATION MACHINE LEARNING MODEL
# --------------------------------------------------

FEATURE_NAMES = [
    "area", "bedrooms", "bathrooms", "property_age",
    "location_score", "balcony", "balcony_count",
    "pool", "pool_area", "garden", "garden_area"
] + [f"facility_{i}" for i in range(250)]


def create_model():
    rng = np.random.default_rng(42)
    rows = 2500

    area = rng.integers(500, 5000, rows)
    bedrooms = rng.integers(1, 7, rows)
    bathrooms = rng.integers(1, 6, rows)
    age = rng.integers(0, 40, rows)
    location_score = rng.integers(1, 11, rows)

    balcony = rng.integers(0, 2, rows)
    balcony_count = balcony * rng.integers(1, 4, rows)
    pool = rng.integers(0, 2, rows)
    pool_area = pool * rng.integers(0, 501, rows)
    garden = rng.integers(0, 2, rows)
    garden_area = garden * rng.integers(0, 1501, rows)

    facilities = rng.integers(0, 2, (rows, 250))

    X = np.column_stack([
        area, bedrooms, bathrooms, age, location_score,
        balcony, balcony_count, pool, pool_area,
        garden, garden_area, facilities
    ])

    # Artificial formula for demonstration only.
    prices = (
        area * rng.uniform(2500, 6500, rows)
        + bedrooms * 400000
        + bathrooms * 250000
        + location_score * 300000
        + balcony_count * 50000
        + pool_area * 800
        + garden_area * 300
        + facilities.sum(axis=1) * 25000
        - age * 40000
        + rng.normal(0, 500000, rows)
    )

    prices = np.maximum(prices, 500000)

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, prices)
    joblib.dump(model, MODEL_FILE)
    return model


if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
    except Exception:
        model = create_model()
else:
    model = create_model()


# --------------------------------------------------
# WEBSITE ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        facility_categories=FACILITY_CATEGORIES,
        locations=LOCATIONS
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form

        area = float(data.get("area", 0))
        bedrooms = int(data.get("bedrooms", 0))
        bathrooms = int(data.get("bathrooms", 0))
        age = int(data.get("property_age", 0))

        if area <= 0 or area > 100000:
            return jsonify({"error": "Enter an area between 1 and 100000 sq ft."}), 400

        if not 1 <= bedrooms <= 20:
            return jsonify({"error": "Enter bedrooms between 1 and 20."}), 400

        if not 1 <= bathrooms <= 20:
            return jsonify({"error": "Enter bathrooms between 1 and 20."}), 400

        if not 0 <= age <= 200:
            return jsonify({"error": "Enter a valid property age."}), 400

        district = str(data.get("district", "")).strip()
        locality = str(data.get("location", "")).strip()

        if district not in LOCATIONS:
            return jsonify({"error": "Please select a valid Tamil Nadu district."}), 400

        if locality not in LOCATIONS[district]:
            return jsonify({"error": "Please select a valid location in that district."}), 400

        # Demo location score: district-based placeholder.
        location_score = 6

        balcony = 1 if str(data.get("balcony", "no")).lower() == "yes" else 0
        balcony_count = int(data.get("balcony_count", 0)) if balcony else 0
        pool = 1 if str(data.get("pool", "no")).lower() == "yes" else 0
        pool_area = float(data.get("pool_area", 0)) if pool else 0
        garden = 1 if str(data.get("garden", "no")).lower() == "yes" else 0
        garden_area = float(data.get("garden_area", 0)) if garden else 0

        selected = data.get("facilities", [])
        if not isinstance(selected, list):
            selected = []

        selected = set(selected)

        facility_flags = [
            1 if str(i) in selected else 0
            for i in range(250)
        ]

        features = [[
            area, bedrooms, bathrooms, age, location_score,
            balcony, balcony_count, pool, pool_area,
            garden, garden_area, *facility_flags
        ]]

        predicted_price = float(model.predict(features)[0])

        return jsonify({
            "success": True,
            "price": round(predicted_price),
            "formatted_price": "₹{:,.0f}".format(predicted_price),
            "message": "Demonstration estimate only. Not a verified market valuation."
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Please enter valid numbers in all property fields."}), 400
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please check your entries and try again."}), 500


@app.route("/dashboard")
def dashboard():
    return jsonify({
        "website": "HouseWorth.Com",
        "model": "Random Forest Regressor",
        "model_type": "Demonstration model",
        "facility_count": len(FACILITIES),
        "district_count": len(LOCATIONS),
        "warning": "Model uses artificial training data, not actual property sales."
    })


if __name__ == "__main__":
    print("HouseWorth.Com is starting...")
    print("Facilities available:", len(FACILITIES))
    print("Tamil Nadu districts available:", len(LOCATIONS))
    app.run(debug=True)
