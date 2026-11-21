
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)

# ---------------- HOUSEWORTH FACILITIES ----------------

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
        "Security Control Room", "Security Alarm Monitoring"
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

FACILITIES = list(dict.fromkeys(
    item for items in FACILITY_CATEGORIES.values() for item in items
))

# ---------------- LOCATIONS ----------------

LOCATIONS = {
   "Ariyalur": [
        "Ariyalur", "Jayankondam", "Sendurai", "Udayarpalayam",
        "Andimadam", "Thirumanur", "Varadarajanpettai",
        "Gangaikondacholapuram", "Meensurutti", "Kallankurichi",
        "Keelapalur", "Vilangudi", "T.Palur", "Ponparappi",
        "Ariyalur Railway Station Area", "Ottakoil", "Sathamangalam",
        "Elakurichi", "Ameenabad", "Kadugur", "Sukkanur",
        "Kallur", "Vettakudi", "Thiruppayur"
    ],

    "Chengalpattu": [
        "Chengalpattu", "Tambaram", "Pallavaram", "Chromepet",
        "Vandalur", "Guduvanchery", "Maraimalai Nagar",
        "Singaperumal Koil", "Kelambakkam", "Thiruporur",
        "Mahabalipuram", "Madurantakam", "Acharapakkam",
        "Chithamur", "Cheyyur", "Thirukazhukundram",
        "Kalpakkam", "Kattankulathur", "Perungalathur",
        "Urapakkam", "Padappai", "Manimangalam", "Otteri",
        "Nandivaram", "Potheri", "Karapakkam", "Sembakkam",
        "Selaiyur", "Perumbakkam", "Siruseri", "Navalur",
        "Semmencheri", "Thazhambur", "Mambakkam", "Ponmar",
        "Kovilambakkam", "Madambakkam", "Pudupakkam",
        "Kandigai", "Kolapakkam", "Thaiyur", "Paiyanur",
        "Nallambakkam", "Vengambakkam", "Rathinamangalam",
        "Moolacheri", "Ottiyambakkam", "Athur", "Pudupattinam"
    ],

    "Chennai": [
        "Adyar", "Anna Nagar", "Ambattur", "Avadi", "Chromepet",
        "Guindy", "Kodambakkam", "Medavakkam", "Mylapore",
        "OMR", "Porur", "T Nagar", "Velachery", "Sholinganallur",
        "Nungambakkam", "Egmore", "Triplicane", "Royapettah",
        "Kilpauk", "Chetpet", "Perambur", "Kolathur", "Purasawalkam",
        "Vyasarpadi", "Tondiarpet", "Washermenpet", "Manali",
        "Ennore", "Thiruvanmiyur", "Besant Nagar", "Pallikaranai",
        "Perungudi", "Thoraipakkam", "Koyambedu", "Mogappair",
        "Ashok Nagar", "KK Nagar", "Saidapet", "Vadapalani",
        "Ayanavaram", "Madipakkam", "Nanganallur", "Pammal",
        "Anakaputhur", "Kundrathur", "Ramapuram", "Alwarpet",
        "Kotturpuram", "Shenoy Nagar", "Raja Annamalaipuram",
        "Puzhal", "Thirumullaivoyal", "Pattabiram", "Teynampet",
        "Thousand Lights", "Park Town", "George Town", "Mint",
        "Choolaimedu", "Aminjikarai", "Arumbakkam", "Saligramam",
        "Virugambakkam", "Valasaravakkam", "Iyyappanthangal",
        "Maduravoyal", "Madhanandapuram", "Manapakkam",
        "Mylai", "Ekkatuthangal", "Jafferkhanpet", "Kottivakkam",
        "Palavakkam", "Neelankarai", "Injambakkam", "Akkarai",
        "Karapakkam", "Perumbakkam", "Sithalapakkam",
        "Vengaivasal", "Sunnambu Kolathur", "Sembakkam",
        "Chitlapakkam", "Hasthinapuram", "Thirumudivakkam",
        "Pozhichalur", "Gerugambakkam", "Moulivakkam",
        "Mangadu", "Thiruverkadu", "Poonamallee", "Nemam",
        "Kattupakkam", "Chembarambakkam", "Thiruninravur",
        "Red Hills", "Minjur", "Ponneri", "Gummidipoondi",
        "Sholavaram", "Tiruverkadu", "Periyapalayam",
        "Kadambathur", "Kakalur", "Manavalanagar", "Veppampattu",
        "Tiruttani", "Uthukottai", "Pakkam", "Thiruvallur"
    ],

    "Coimbatore": [
        "Coimbatore", "Gandhipuram", "RS Puram", "Peelamedu",
        "Saravanampatti", "Vadavalli", "Singanallur",
        "Saibaba Colony", "Thudiyalur", "Thondamuthur",
        "Theethipalayam", "Pollachi", "Mettupalayam",
        "Madampatti", "Alandurai", "Alanthurai", "Semmedu",
        "Devarayapuram", "Narasipuram", "Pooluvampatti",
        "Ikkarai Boluvampatti", "Sundakkamuthur", "Thennamanallur",
        "Veerakeralam", "Vedapatti", "Perur", "Perur Chettipalayam",
        "West Chitrachavadi", "East Chitrachavadi",
        "Kalikkanaickenpalayam", "Jakirnaickenpalayam",
        "Komarapalayam", "Madvarayapuram", "Dhaliyur",
        "Thenkarai", "Pannimadai", "Somayampalayam",
        "Chinna Thadagam", "Anaikatti", "Kavundampalayam",
        "Periyanaickenpalayam", "Naickenpalayam", "Bilichi",
        "Veerapandi", "Kurudampalayam", "Nanjundapuram",
        "Telungupalayam", "Puliakulam", "Krishnarayapuram",
        "Sanganur", "Chinnavedampatti", "Vilankurichi",
        "Kalapatti", "Keeranatham", "Kovilpalayam",
        "Sarcarasamakulam", "Agraharasamakulam", "Idigarai",
        "Annur", "Annur Mettupalayam", "Kanjampalli",
        "Kariyampalayam", "Kuppanur", "Kuppepalayam",
        "Allapalayam", "Akkaraisengapalli", "Ambothi",
        "Arachipalayam", "Masagoundenchettipalayam",
        "Naranapuram", "Odderpalayam", "Pachapalayam",
        "Pasur", "Pillayampalayam", "Vadakkalur",
        "Vellanaipatti", "Marudur", "Sikkarampalayam",
        "Sirumugai", "Karamadai", "Odanthurai", "Nellithurai",
        "Thekkampatti", "Tholampalayam", "Bellepalayam",
        "Kalampalayam", "Kemmarampalayam", "Kallar",
        "Kottaipalayam", "Vellalore", "Podanur", "Kurichi",
        "Kuniyamuthur", "Kovaipudur", "Madukkarai",
        "Ettimadai", "Pichanur", "Palathurai",
        "Malumichampatti", "Seerapalayam", "Thirumalayampalayam",
        "Valukkuparai", "Othakkalmandapam", "Nachipalayam",
        "Arisipalayam", "Kinathukadavu", "Solavampalayam",
        "Vadapudur", "Kuthirayalampalayam", "Arasampalayam",
        "Pottaiyandipurambu", "Panapatti", "Vadasithur",
        "Devanampalayam", "Chettipalayam", "Ottakkalmandapam",
        "Samathur", "Perianegamam", "Zamin Uthukuli",
        "Vettaikaranpudur", "Kottur", "Dhalavaipalayam",
        "Valparai", "Anaimalai", "Aliyar Nagar",
        "Ambarampalayam", "Angalakurichi", "A.Nagoor",
        "Andipalayam", "Achipatti", "Akilandapuram",
        "Alagirichettipalayam", "Chinnampalayam", "Gomangalam",
        "Gomangalam Pudhur", "Jallipatti", "Kanjampatti",
        "Makkinaickenpatti", "Nallur", "Negamam", "Puliampatti",
        "Rasakkapalayam", "Suleeswaranpatti", "Thippampatti",
        "Udumalai Road", "Vellamalai", "Sultanpet",
        "Karumathampatti", "Irugur", "Neelambur", "Arasur",
        "Kannampalayam", "Sulur", "Kangeyampalayam",
        "Kalangal", "Kallapalayam", "Kadampadi", "Pattanam",
        "Peedampalli", "Pappampatti", "Rasipalayam",
        "Mylampatti", "Bogampatti", "Appanaickenpatti",
        "Senjeriputhur", "Kumarapalayam", "Giddampalayam",
        "Selakkarichal", "Mopperipalayam", "N.S.N. Palayam",
        "Pallapalayam", "Pooluvapatti", "Periyanaickenpalayam",
        "Idayarpalayam", "Kanuvakarai", "Kattampatti",
        "Vellamadai", "Kondaiyampalayam", "Kunnathur",
        "Masakkal", "Ranganathapuram", "Thennampalayam",
        "Karamadai Road", "Puliakulam Road", "Race Course",
        "Town Hall", "Ukkadam", "Ramanathapuram", "Trichy Road",
        "Avinashi Road", "Mettupalayam Road", "Pollachi Road",
        "Palakkad Road", "Siruvani Road", "Marudhamalai Road",
        "Kalapatti Road", "Vilankurichi Road", "Saravanampatti Road",
        "Eachanari", "Ondipudur", "Chinniampalayam",
        "Sundarapuram", "NGGO Colony", "Thudiyalur",
        "Vadavalli", "Thondamuthur", "Karunya Nagar",
        "Narasipuram", "Semmedu", "Theethipalayam",
        "Devarayapuram", "Sundakkamuthur", "Pooluvampatti",
        "Jakirnaickenpalayam", "Kalikkanaickenpalayam",
        "Komarapalayam", "Madvarayapuram", "Vellimalaipattinam",
        "Narasimhanaickenpalayam", "Naickenpalayam", "Bilichi",
        "Kurudampalayam", "Pannimadai", "Chinna Thadagam",
        "Somayampalayam", "Kuppanur", "Kunnathur",
        "Mylampatti", "Otterpalayam", "Pappampatti", "Pattanam",
        "Selakkarichel", "Semmandampalayam", "Appanayakkanpatti",
        "Giddampalayam", "Senjeriputhur", "Kumarapalayam"
    ],

    "Cuddalore": [
        "Cuddalore", "Chidambaram", "Panruti", "Virudhachalam",
        "Neyveli", "Bhuvanagiri", "Kattumannarkoil",
        "Kurinjipadi", "Parangipettai", "Annamalai Nagar",
        "Sethiathoppu", "Thittakudi", "Srimushnam", "Vadalur",
        "Killai", "Nellikuppam", "Melpattampakkam",
        "Kumaratchi", "Kattumannarkudi", "Mangalampet",
        "Ramanatham", "Veppur", "Pennadam", "Tittakudi",
        "Thiruvandipuram", "Pathirikuppam", "Thiruvathigai",
        "Kadampuliyur", "Marungur", "Ayyampettai", "Alapakkam",
        "Puduchathiram", "Kothattai", "Sathapadi", "Ariyankuppam"
    ],

    "Dharmapuri": [
        "Dharmapuri", "Harur", "Palacode", "Pennagaram",
        "Pappireddipatti", "Nallampalli", "Karimangalam",
        "Morappur", "Bommidi", "Kadathur", "Marandahalli",
        "Indur", "Adhiyamankottai", "Hogenakkal",
        "Eriyur", "Koothapadi", "Mallapuram", "Kambainallur",
        "Doddampatti", "Thoppur", "Sogathur", "Elakkiyampatti",
        "Vellakuttai", "Madhikonpalayam", "Ayyampatti",
        "Pochampalli Road", "Kottapatti", "Mookkanur",
        "Neruppur", "Somanahalli", "Gopalapuram"
    ],

    "Dindigul": [
        "Dindigul", "Palani", "Kodaikanal", "Oddanchatram",
        "Vedasandur", "Nilakottai", "Natham", "Batlagundu",
        "Ayyalur", "Chinnalapatti", "Guziliamparai",
        "Kallimandayam", "Sempatti", "Thadikombu",
        "Athoor", "Sithayankottai", "Pannaikadu",
        "Pallapatti", "Kannivadi", "Reddiyarchatram",
        "Neikkarapatti", "Balasamudram", "Ayakudi",
        "Keeranur", "Kamarajapuram", "Ammaiyappanur",
        "Kottur", "Sanarpatti", "Mettupatti", "Vadamadurai",
        "Kovilur", "Ambathurai", "Thoppampatti",
        "Kondamanayakkanpatti", "Pudukottai", "Gopalpatti"
    ],

    "Erode": [
        "Erode", "Perundurai", "Bhavani", "Gobichettipalayam",
        "Sathyamangalam", "Anthiyur", "Chennimalai", "Modakurichi",
        "Kodumudi", "Nambiyur", "Thalavadi", "Arachalur",
        "Kavindapadi", "Sivagiri", "Vellakoil", "Mylambadi",
        "Elumathur", "Bannari", "Punjai Puliampatti",
        "Appakudal", "Kasipalayam", "Kanjikoil", "Nasiyanur",
        "Periyasemur", "Surampatti", "Veerappanchatram",
        "Thindal", "Lakkapuram", "Chithode", "Mettukadai",
        "Pallipalayam", "Komarapalayam", "Nerunjipettai",
        "Ammapettai", "Vellithiruppur", "Bhavanisagar",
        "Goundampalayam", "Kembanaickenpalayam", "Thingalur",
        "Vijayamangalam", "Ingur", "Pethampalayam",
        "Kanjampalayam", "Perundurai Road", "Mettupalayam"
    ],

    "Kallakurichi": [
        "Kallakurichi", "Chinnasalem", "Sankarapuram",
        "Tirukoilur", "Ulundurpet", "Kalvarayan Hills",
        "Rishivandiyam", "Thiyagadurgam", "Manalurpet",
        "Vadakkanandal", "Sulankurichi", "Elavanasur",
        "Siruvangur", "Moongilthuraipattu", "Pinnalvadi",
        "Kachirayapalayam", "Ariyalur", "Thottiyam",
        "Pethanur", "Pallagacheri", "Pukkiravari",
        "Vellimalai", "Kaniyamur", "Madur", "Nainarpalayam",
        "Virugavur", "Peruvangur", "Vasudevanur"
    ],

    "Kancheepuram": [
        "Kancheepuram", "Sriperumbudur", "Walajabad",
        "Uthiramerur", "Kundrathur", "Oragadam",
        "Sunguvarchatram", "Kavithandalam", "Padappai",
        "Thirumazhisai", "Enathur", "Ayyampettai",
        "Magaral", "Konerikuppam", "Pillaipakkam",
        "Manimangalam", "Vallam", "Irungattukottai",
        "Mambakkam", "Thirupandiyur", "Thirumukudal",
        "Nathapettai", "Pazhayaseevaram", "Kaliyampoondi",
        "Vaiyavur", "Siruveliyur", "Perunagar", "Kavanipakkam",
        "Kancheepuram Town", "Poonjeri", "Kottavakkam",
        "Kaliyur", "Ayyangarkulam", "Kazhiyur"
    ],

    "Kanniyakumari": [
        "Nagercoil", "Kanyakumari", "Marthandam", "Colachel",
        "Thuckalay", "Padmanabhapuram", "Kulasekaram",
        "Karungal", "Eraniel", "Kuzhithurai", "Thingal Nagar",
        "Suchindram", "Aralvaimozhi", "Boothapandi",
        "Thiruvattar", "Vilavancode", "Manavalakurichi",
        "Puthukadai", "Kottaram", "Parvathipuram",
        "Vadasery", "Asaripallam", "Kottar", "Chettikulam",
        "Thirparappu", "Pechiparai", "Mondaikadu",
        "Kaliyakkavilai", "Palliyadi", "Azhagappapuram",
        "Rajakkamangalam", "Muttom", "Kanyakumari Town",
        "Kappukadu", "Thirunainarkurichi", "Kumarapuram",
        "Ezhudesam", "Kanniyakumari District"
    ],

    "Karur": [
        "Karur", "Kulithalai", "Krishnarayapuram", "Aravakurichi",
        "Thanthoni", "Pugalur", "Vangal", "Nerur",
        "Kadavur", "Pallapatti", "Chinnadharapuram",
        "Velayuthampalayam", "Mayanur", "Lalapet",
        "Manmangalam", "Thogamalai", "Vennaimalai",
        "Karuppur", "Noyyal", "Panjapatti", "K.Paramathi",
        "Punjaipugalur", "Nangunur", "Achimangalam",
        "Sukaliyur", "Ramakrishnapuram", "Pasupathipalayam",
        "Sanapiratti", "Koyamballi", "Velliyanai"
    ],

    "Krishnagiri": [
        "Krishnagiri", "Hosur", "Denkanikottai", "Bargur",
        "Pochampalli", "Uthangarai", "Shoolagiri", "Kelamangalam",
        "Kaveripattinam", "Mathur", "Veppanapalli",
        "Anchetty", "Rayakottai", "Bagalur", "Moranapalli",
        "Thally", "Berigai", "Chandrapuram", "Kundarapalli",
        "Kakkadasam", "Anekal Road", "Kattiganapalli",
        "Singarapettai", "Samalpatti", "Karapattu",
        "Marasandiram", "Perandapalli", "Mugalur",
        "Alur", "Nallur", "Soolagiri", "Hosur SIPCOT",
        "Avalapalli", "Gurubarapalli", "Bukkasagaram"
    ],

    "Madurai": [
        "Madurai", "Thirumangalam", "Usilampatti", "Melur",
        "Vadipatti", "Alanganallur", "Sholavandan", "Peraiyur",
        "Kallupatti", "Tirupparankundram", "Avaniyapuram",
        "Anaiyur", "Othakadai", "Yanaimalai", "Samayanallur",
        "Paravai", "Nagamalai Pudukottai", "Kappalur",
        "Silaiman", "Karuppayurani", "Thiruvedagam",
        "Goripalayam", "Tallakulam", "Simmakkal", "Arapalayam",
        "Koodal Nagar", "Pudur", "Mattuthavani", "Kalavasal",
        "Bypass Road", "Anna Nagar", "Vilangudi", "Jaihindpuram",
        "Thirunagar", "Pasumalai", "Villapuram", "Sathamangalam",
        "Narimedu", "Kochadai", "Chokkikulam", "Surya Nagar",
        "Iyer Bungalow", "Kallanthiri", "Pudur", "Thiruppalai",
        "Ayyar Bangalow", "Vandiyur", "Kamarajar Salai"
    ],

    "Mayiladuthurai": [
        "Mayiladuthurai", "Sirkazhi", "Tharangambadi",
        "Kuthalam", "Vaitheeswarankoil", "Manalmedu",
        "Porayar", "Kollidam", "Poompuhar", "Akkur",
        "Thirukadaiyur", "Kuttalam", "Elanthangudi",
        "Vanjur", "Nallur", "Moovalur", "Koranad",
        "Pandur", "Mappadugai", "Kiliyanur", "Kanjanagaram",
        "Akkur", "Thiruvengadu", "Sembanar Koil",
        "Melaiyur", "Karuva", "Nidur", "Manjolai"
    ],

    "Nagapattinam": [
        "Nagapattinam", "Velankanni", "Kilvelur", "Thirukkuvalai",
        "Vedaranyam", "Thalainayar", "Akkaraipettai",
        "Sikkal", "Thirumarugal", "Kizhvelur", "Nagore",
        "Seruthur", "Kariyapattinam", "Kodiyakarai",
        "Manjakollai", "Vadakku Poigainallur", "Akkaraipul",
        "Poravacheri", "Thevur", "Pappakoil", "Sembiyanmahadevi",
        "Thittacheri", "Moolakudi", "Kurukathi", "Nagai Town",
        "Vailankanni Road", "Puducherry Border Area"
    ],

    "Namakkal": [
        "Namakkal", "Rasipuram", "Tiruchengode", "Paramathi Velur",
        "Kumarapalayam", "Mohanur", "Sendamangalam", "Erumaipatti",
        "Pallipalayam", "Mallasamudram", "Elachipalayam",
        "Kolli Hills", "Jedarpalayam", "Vennandur",
        "Nallipalayam", "Anangur", "Puduchatram", "Kalangani",
        "Namagiripettai", "Belukurichi", "Kabilarmalai",
        "Pothanur", "Velur", "Pandamangalam", "Seerappalli",
        "Muthugapatti", "Thathaiyangarpettai", "Athanur",
        "Kandampalayam", "Koneripatti", "Nallur"
    ],

    "Nilgiris": [
        "Ooty", "Coonoor", "Kotagiri", "Gudalur", "Pandalur",
        "Ketti", "Lovedale", "Wellington", "Aravankadu",
        "Kundah", "Manjoor", "Emerald", "Avalanche",
        "Doddabetta", "Naduvattam", "Devala", "Masinagudi",
        "Sholur", "Adigaratty", "Fernhill", "Charing Cross",
        "Upper Bazaar", "Fingerpost", "Theetukal", "Thalakunda",
        "Muthorai", "Hulical", "Burliar", "Yedapalli",
        "Bikkapathi", "Cherangode", "Ayyankolli", "Mavanalla",
        "Nelliyalam", "Pykara", "Glenmorgan", "Kattabettu",
        "Rangasamy Peak", "Kodanad", "Longwood"
    ],

    "Perambalur": [
        "Perambalur", "Veppanthattai", "Kunnam", "Alathur",
        "Chettikulam", "Labbaikudikadu", "Padalur",
        "Eraiyur", "Siruvachur", "Esanai", "Nattarmangalam",
        "Ammapalayam", "Vellar", "Anukkur", "Arumbavur",
        "Vasistapuram", "Pudunadu", "Thuraimangalam",
        "Kalarampatti", "Pudhu Vettakudi", "Kollai",
        "Kunnam Road", "Sathanur", "Melamathur", "Vayalur",
        "Annamangalam", "Pachaiyamman Nagar"
    ],

    "Pudukkottai": [
        "Pudukkottai", "Aranthangi", "Alangudi", "Keeranur",
        "Avudaiyarkoil", "Gandarvakottai", "Thirumayam",
        "Viralimalai", "Manamelkudi", "Ponnamaravathi",
        "Karambakudi", "Annavasal", "Kottaipattinam",
        "Narthamalai", "Illuppur", "Kattumavadi", "Mimisal",
        "Aavur", "Kudimiyanmalai", "Viralimalai", "Rangiyam",
        "Thirukokarnam", "Kavinadu", "Arimalam", "Perungalur",
        "Mullur", "Vellanur", "Pudhuvayal", "Karaikudi Border",
        "Pudukkottai Town", "Pudukkottai Collectorate Area"
    ],

    "Ramanathapuram": [
        "Ramanathapuram", "Rameswaram", "Paramakudi",
        "Kamuthi", "Mudukulathur", "Thiruvadanai",
        "R.S. Mangalam", "Sayalkudi", "Mandapam", "Pamban",
        "Kilakarai", "Uchipuli", "Abiramam", "Sakkarakottai",
        "Devipattinam", "Ariyankundu", "Valinokkam",
        "Thondi", "Nainarkoil", "Chathirakudi", "Perunali",
        "Kadaladi", "Vallandai", "Sundarapandian Pattinam",
        "Mookaiyur", "Narippaiyur", "Ervadi", "Sathirakudi",
        "Vani", "Mandalamanickam", "Panaikulam"
    ],

    "Ranipet": [
        "Ranipet", "Arcot", "Walajapet", "Sholingur",
        "Arakkonam", "Nemili", "Kalavai", "Melvisharam",
        "Timiri", "Vilapakkam", "Thakkolam", "Ammur",
        "Kaveripakkam", "Banavaram", "Muthukadai",
        "Navalpur", "Rathinagiri", "Vannivedu", "Lalapet",
        "Thiruvalam", "Ponnai", "Puliyangannu", "Sumaithangi",
        "Ariyur", "Sengadu", "Kondapalayam", "Sathambakkam",
        "Bharathidasan Nagar", "Ranipet SIPCOT", "BHEL Township"
    ],

    "Salem": [
        "Salem", "Attur", "Mettur", "Omalur", "Edappadi",
        "Tharamangalam", "Sankari", "Yercaud", "Gangavalli",
        "Valapady", "Nangavalli", "Mecheri", "Kolathur",
        "Kadayampatti", "Pethanaickenpalayam", "Veerapandi",
        "Ayothiapattinam", "Karuppur", "Panamarathupatti",
        "Kondalampatti", "Hasthampatti", "Ammapet", "Fairlands",
        "Five Roads", "Alagapuram", "Suramangalam", "Kannankurichi",
        "Mullai Nagar", "Shevapet", "Gugai", "Dadagapatti",
        "Junction", "Leigh Bazaar", "Kondappanaickenpatti",
        "Ponnammapet", "Reddipatti", "Sivathapuram",
        "Thirumalaigiri", "Kitchipalayam", "Narasothipatti",
        "Mallur", "Magudanchavadi", "Konganapuram", "Poolampatti",
        "Mettunaickenpatti", "Elampillai", "Veeranam"
    ],

    "Sivaganga": [
        "Sivaganga", "Karaikudi", "Devakottai", "Manamadurai",
        "Tiruppattur", "Ilayangudi", "Singampunari", "Kalaiyarkoil",
        "Thirupuvanam", "Sakkottai", "Kallal", "Kanadukathan",
        "Pallathur", "Kandanur", "Nattarasankottai",
        "Kottaiyur", "S.Pudur", "Vairavanpatti", "Puduvayal",
        "Ariyakudi", "Karaikudi New Town", "Mithilaipatti",
        "Kandavarayanpatti", "Soorakkudi", "Piranmalai",
        "Koviloor", "Pillayarpatti", "Nachiapuram",
        "Kallupatti", "Thiruppachethi", "Okkur"
    ],

    "Tenkasi": [
        "Tenkasi", "Sankarankovil", "Kadayanallur", "Puliyangudi",
        "Courtallam", "Shenkottai", "Alangulam", "Vasudevanallur",
        "Surandai", "Pavoorchatram", "Sivagiri", "Ayikudi",
        "Kadaiyanallur", "Thirumalapuram", "Sundarapandiapuram",
        "Panpoli", "Mekkarai", "Achanpudur", "Kuthukalvalasai",
        "Ilathur", "Sambavar Vadakarai", "Sinthamani",
        "Kallur", "Pudur", "Vadiyur", "Thiruvengadam",
        "Kuruvikulam", "Nelkattumseval", "Vadakarai",
        "Vallam", "Karisalkulam"
    ],

    "Thanjavur": [
        "Thanjavur", "Kumbakonam", "Pattukkottai", "Peravurani",
        "Orathanadu", "Thiruvaiyaru", "Papanasam", "Thiruvidaimarudur",
        "Ayyampettai", "Adirampattinam", "Vallam", "Ammapettai",
        "Madukkur", "Sengipatti", "Swamimalai", "Nachiyar Koil",
        "Thirukattupalli", "Budalur", "Kandiyur", "Melattur",
        "Tiruvonam", "Kumbakonam Town", "Darasuram", "Thirubuvanam",
        "Thiruppanandal", "Sundaraperumal Koil", "Veppathur",
        "Kabisthalam", "Pasupathikoil", "Rajagiri", "Papanasam Road",
        "Saliyamangalam", "Kovilvenni", "Naducauvery",
        "Vennar Bank", "Alangudi", "Mannargudi Road"
    ],

    "Theni": [
        "Theni", "Periyakulam", "Bodinayakanur", "Cumbum",
        "Uthamapalayam", "Andipatti", "Chinnamanur", "Gudalur",
        "Kombai", "Thevaram", "Myladumparai", "Veerapandi",
        "Kadamalaikundu", "Devadanapatti", "Silamalai",
        "Palani Chettipatti", "Sankarapuram", "Kottur",
        "Bala Krishnapuram", "Allinagaram", "Lakshmipuram",
        "Aundipatti", "Kattabomman Street", "Bodi Mettu",
        "Kottagudi", "Sothuparai", "Vaigai Dam", "Gandamanur",
        "Ramakrishnapuram", "Jeyamangalam", "Vadugapatti"
    ],

    "Thoothukudi": [
        "Thoothukudi", "Kovilpatti", "Tiruchendur", "Srivaikuntam",
        "Sathankulam", "Ettayapuram", "Vilathikulam",
        "Ottapidaram", "Kayathar", "Arumuganeri", "Nazareth",
        "Alwarthirunagari", "Pudur", "Kalugumalai",
        "Authoor", "Kulathur", "Pattinam", "Vembar",
        "Mullakkadu", "Tharuvaikulam", "Kayalpattinam",
        "Pudukottai", "Mappilaiurani", "Korampallam",
        "Meganapuram", "Kadambur", "Maniyachi", "Pasuvanthanai",
        "Kazhugumalai", "Sankaralingapuram", "Vilathikulam",
        "Eral", "Ayyanadaippu", "Tiruppani"
    ],

    "Tiruchirappalli": [
        "Tiruchirappalli", "Srirangam", "Thiruverumbur",
        "Manapparai", "Lalgudi", "Musiri", "Thuraiyur",
        "Manachanallur", "Samayapuram", "Kattur", "Ponmalai",
        "Kollidam", "Vayalur", "Pettavaithalai", "Sirugambur",
        "Inamkulathur", "Bikshandarkoil", "Uyyakondan Thirumalai",
        "Edamalaipatti Pudur", "Navalpattu", "Palakarai",
        "Woraiyur", "Kambarasampettai", "Ariyamangalam",
        "Cantonment", "Thillai Nagar", "KK Nagar", "Puthur",
        "Thennur", "Vasantha Nagar", "Airport Area", "TVS Tollgate",
        "Karumandapam", "Subramaniapuram", "Mannarpuram",
        "Kallukuzhi", "Panjapur", "Sirugamani", "Valadi",
        "Poongudi", "Tiruvarambur", "BHEL Township"
    ],

    "Tirunelveli": [
        "Tirunelveli", "Palayamkottai", "Ambasamudram",
        "Nanguneri", "Cheranmahadevi", "Valliyur", "Radhapuram",
        "Kalakad", "Thisayanvilai", "Manur", "Pettai",
        "Melapalayam", "Thachanallur", "Gangaikondan",
        "Kallur", "Moolaikarakurichi", "Eruvadi", "Panagudi",
        "Tirunelveli Junction", "Vannarpettai", "Maharaja Nagar",
        "High Ground", "Perumalpuram", "Sankar Nagar",
        "Thalaiyuthu", "Suthamalli", "Nallur", "Muneerpallam",
        "Kalakkad", "Kavalkinaru", "Azhagiyapandiapuram",
        "Kalakkadu Road", "Kallidaikurichi", "Veeravanallur",
        "Papanasam", "Manimuthar", "Tirukurungudi"
    ],

    "Tirupathur": [
        "Tirupathur", "Vaniyambadi", "Ambur", "Natrampalli",
        "Jolarpet", "Alangayam", "Kandili", "Pudurnadu",
        "Yelagiri", "Madhanur", "Kothur", "Pachal",
        "Achamangalam", "Bommikuppam", "Pernampattu Road",
        "Yelagiri Hills", "Athiyur", "Sundarampalli",
        "Vellakuttai", "Kethandapatti", "Kurumbarapatti",
        "Kakkanampalayam", "Pudur", "Kothapalli",
        "Chinnaveppambattu", "Nimmiyampattu", "Kadirimangalam",
        "Mottur", "Mallagunda", "Kollapatti"
    ],

    "Tiruppur": [
        "Tiruppur", "Avinashi", "Palladam", "Udumalpet",
        "Dharapuram", "Kangeyam", "Uthukuli", "Madathukulam",
        "Mulanur", "Thirumuruganpoondi", "Nallur", "Mangalam",
        "Anupparpalayam", "Velampalayam", "Perumanallur",
        "Chettipalayam", "Neripperichal", "Vijayapuram",
        "Kumar Nagar", "P.N. Road", "Gandhi Nagar",
        "Kovilvazhi", "Veerapandi", "Pallapalayam",
        "Rakiyapalayam", "Sivanmalai", "Chinnakandi",
        "Kangeyam Road", "Perichipalayam", "Mettupalayam",
        "Palladam Road", "Noyyal", "Mangalam Road",
        "Nallur", "Kallangadu", "K Chettipalayam",
        "Velliyangadu", "Kumarapalayam", "Thottipalayam",
        "Ayyampalayam", "Samarathottam", "Vavipalayam",
        "Karanampettai", "Arulpuram", "Pongalur",
        "Pethampalayam", "Avinashi Road", "Kunnathur",
        "Perumanallur Road", "Nambiyur Road", "Kudimangalam"
    ],

    "Tiruvallur": [
        "Tiruvallur", "Poonamallee", "Tiruttani", "Gummidipoondi",
        "Ponneri", "Uthukottai", "Avadi", "Ambattur",
        "Pattabiram", "Thiruninravur", "Minjur", "Red Hills",
        "Sholavaram", "Kadambathur", "Periyapalayam",
        "Kakalur", "Manavalanagar", "Veppampattu",
        "Pakkam", "Tiruverkadu", "Nemam", "Thirumazhisai",
        "Vellavedu", "Poondi", "Kondancherry", "Egattur",
        "Thiruvallur Town", "Sevvapet", "Putlur", "Aranvoyal",
        "Kollur", "Satharai", "Kakkalur Industrial Estate",
        "Manali New Town", "Velliyur", "Puduvoyal",
        "Periyapalayam Road", "Nallur", "Elavur"
    ],

    "Tiruvannamalai": [
        "Tiruvannamalai", "Arani", "Polur", "Cheyyar",
        "Chengam", "Vandavasi", "Kalasapakkam", "Thandrampet",
        "Jamunamarathur", "Kilpennathur", "Chetpet",
        "Kannamangalam", "Sathanur", "Desur", "Peranamallur",
        "Pudupalayam", "Vettavalam", "Sembedu", "Adaiyur",
        "Thiruvannamalai Town", "Tiruvannamalai Girivalam Road",
        "Mathur", "Mangalampadi", "Kadaladi", "Nammiyandal",
        "Somasipadi", "Vazhavachanur", "Kanniya", "Nambiyandal",
        "Kalambur", "Pavithram", "Arunachalam Nagar",
        "Kilnathur", "Samudram", "Thachampattu", "Vengikkal",
        "Nallavanpalayam", "Pavupattu", "Kattampoondi"
    ],

    "Tiruvarur": [
        "Tiruvarur", "Mannargudi", "Thiruthuraipoondi",
        "Needamangalam", "Nannilam", "Valangaiman",
        "Koothanallur", "Kodavasal", "Muthupet", "Koradacheri",
        "Alangudi", "Kudavasal", "Adiyakkamangalam",
        "Thirukannamangai", "Peralam", "Kottur", "Thirumakkottai",
        "Paravakottai", "Sundarakkottai", "Vilamal",
        "Pulivalam", "Thirunellikaval", "Kamalapuram",
        "Mannargudi Town", "Serukalathur", "Thiruvathangudi",
        "Kappanamangalam", "Haridwaramangalam", "Thirukkalar",
        "Thiruvanduthurai", "Kottarakudi"
    ],

    "Vellore": [
        "Vellore", "Katpadi", "Gudiyatham", "Pernambut",
        "Anaicut", "K.V. Kuppam", "Bagayam", "Virudhambattu",
        "Sathuvachari", "Thorapadi", "Melmonavur", "Odugathur",
        "Latteri", "Pennathur", "Pallikonda", "Virinchipuram",
        "Ussoor", "Kannamangalam", "Adukkamparai",
        "Vellore Fort Area", "Green Circle", "Christian Medical College Area",
        "New Bus Stand Area", "Katpadi Railway Station Area",
        "Sainathapuram", "Kosapet", "Allapuram", "Bagayam",
        "Rangapuram", "Konavattam", "Sathuvachari",
        "Vellapadi", "Ariyur", "Melvisharam Road",
        "Kalinjur", "Pallikuppam", "Vellore Town"
    ],

    "Viluppuram": [
        "Viluppuram", "Tindivanam", "Gingee", "Marakkanam",
        "Vikravandi", "Vanur", "Kandamangalam", "Koliyanur",
        "Valavanur", "Thiruvennainallur", "Arakandanallur",
        "Mailam", "Mugaiyur", "Ananthapuram", "Kedar",
        "Panayapuram", "Salavadi", "Nemur", "Radhapuram",
        "Thirunavalur", "Ulundurpet Road", "Kandachipuram",
        "Avalurpet", "Melmalayanur", "Siruvadi", "Ayyur",
        "Kandamangalam", "Perumbakkam", "Kiliyanur",
        "Vellimedupettai", "Thiruvamathur", "Pakkam",
        "Kottakuppam", "Poothurai", "Auroville Road",
        "Kottampakkam", "Puducherry Border", "Vettavalam Road"
    ],

    "Virudhunagar": [
        "Virudhunagar", "Sivakasi", "Aruppukottai", "Rajapalayam",
        "Srivilliputhur", "Sattur", "Kariapatti", "Vembakottai",
        "Watrap", "Tiruchuli", "Narikudi", "Mamsapuram",
        "Seithur", "Chatrapatti", "Alangulam", "Pudupatti",
        "Mallankinaru", "Thalavaipuram", "Kovilpatti Road",
        "Sundarapandiam", "W. Pudupatti", "Krishnapuram",
        "Devadanam", "Samusigapuram", "Shenbagathoppu",
        "Aruppukottai Town", "Thiruthangal", "Anaiyur",
        "Vijayakarisalkulam", "Kanjampatti", "Kalkurichi",
        "Vadamalaikurichi", "Kallur", "Pudhu Natham",
        "Sathuragiri Road", "Mullikulam", "Karisalkulam"
    ]
}

# ---------------- DEMO AI MODEL ----------------

FEATURE_NAMES = [
    "area", "bedrooms", "bathrooms", "property_age",
    "location_score", "balcony", "balcony_count",
    "pool", "pool_area", "garden", "garden_area"
] + FACILITIES

def create_model():
    rng = np.random.default_rng(42)
    rows = 1500

    area = rng.uniform(500, 6000, rows)
    bedrooms = rng.integers(1, 7, rows)
    bathrooms = rng.integers(1, 6, rows)
    age = rng.integers(0, 40, rows)
    location_score = rng.uniform(1, 10, rows)
    balcony = rng.integers(0, 2, rows)
    balcony_count = rng.integers(0, 5, rows)
    pool = rng.integers(0, 2, rows)
    pool_area = rng.uniform(0, 1000, rows) * pool
    garden = rng.integers(0, 2, rows)
    garden_area = rng.uniform(0, 2000, rows) * garden
    facilities = rng.integers(0, 2, (rows, len(FACILITIES)))

    X = np.column_stack([
        area, bedrooms, bathrooms, age, location_score,
        balcony, balcony_count, pool, pool_area,
        garden, garden_area, facilities
    ])

    y = (
        area * 4000 + bedrooms * 350000
        + bathrooms * 200000 + location_score * 200000
        + balcony_count * 50000 + pool_area * 500
        + garden_area * 200 + facilities.sum(axis=1) * 20000
        - age * 30000 + rng.normal(0, 400000, rows)
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = create_model()

# ---------------- ROUTES ----------------

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
        data = request.get_json(force=True)

        area = float(data.get("area", 0))
        bedrooms = int(data.get("bedrooms", 0))
        bathrooms = int(data.get("bathrooms", 0))
        property_age = int(data.get("property_age", 0))

        district = data.get("district", "")
        location = data.get("location", "")

        if district not in LOCATIONS or location not in LOCATIONS[district]:
            return jsonify({"error": "Please select a valid district and locality."}), 400

        if not (100 <= area <= 100000):
            return jsonify({"error": "Area must be between 100 and 100000 sqft."}), 400

        balcony = 1 if str(data.get("balcony", "no")).strip().lower() == "yes" else 0
        balcony_count = int(data.get("balcony_count", 0))
        pool = 1 if str(data.get("pool", "no")).strip().lower() == "yes" else 0
        pool_area = float(data.get("pool_area", 0))
        garden = 1 if str(data.get("garden", "no")).strip().lower() == "yes" else 0
        garden_area = float(data.get("garden_area", 0))

        selected = set(str(x) for x in data.get("facilities", []))
        facilities = [
            1 if str(i) in selected else 0 for i in range(len(FACILITIES))
        ]

        features = np.array([[
            area, bedrooms, bathrooms, property_age,
            6, balcony, balcony_count, pool, pool_area,
            garden, garden_area, *facilities
        ]])

        price = float(model.predict(features)[0])

        return jsonify({
            "success": True,
            "price": price,
            "formatted_price": "₹{:,.0f}".format(max(0, price)),
            "message": "Demonstration estimate only; not based on verified property sales."
        })

    except (ValueError, TypeError):
        app.logger.exception("Invalid numeric input in prediction form")
        return jsonify({"error": "Invalid numeric input. Check Render logs."}), 400

    except Exception:
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
    print("HouseWorth.Com starting...")
    print("Facility options:", len(FACILITIES))
    print("Districts:", len(LOCATIONS))
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
