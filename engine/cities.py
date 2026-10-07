"""Built-in gazetteer of Indian cities, so 'a city the directory does not cover' is detected
deterministically (no LLM needed). Maps lower-case name or alias -> canonical name."""
import re

_CANON = {
    "Mumbai": ["bombay"], "Delhi": ["new delhi"], "Bengaluru": ["bangalore"], "Hyderabad": [],
    "Ahmedabad": [], "Chennai": ["madras"], "Kolkata": ["calcutta"], "Surat": [], "Pune": ["poona"],
    "Jaipur": [], "Lucknow": [], "Kanpur": [], "Nagpur": [], "Indore": [], "Thane": [], "Bhopal": [],
    "Visakhapatnam": ["vizag"], "Patna": [], "Vadodara": ["baroda"], "Ghaziabad": [], "Ludhiana": [],
    "Agra": [], "Nashik": [], "Faridabad": [], "Meerut": [], "Rajkot": [], "Varanasi": ["banaras"],
    "Srinagar": [], "Aurangabad": [], "Amritsar": [], "Navi Mumbai": [], "Prayagraj": ["allahabad"],
    "Ranchi": [], "Howrah": [], "Coimbatore": [], "Jabalpur": [], "Gwalior": [], "Vijayawada": [],
    "Jodhpur": [], "Madurai": [], "Raipur": [], "Guwahati": [], "Chandigarh": [], "Mysuru": ["mysore"],
    "Gurugram": ["gurgaon"], "Noida": [], "Thiruvananthapuram": ["trivandrum"], "Kochi": ["cochin"],
    "Bhubaneswar": [], "Dehradun": [], "Mangaluru": ["mangalore"], "Jalandhar": [],
    "Tiruchirappalli": ["trichy"], "Udaipur": [], "Siliguri": [], "Jammu": [], "Goa": [],
}
GAZETTEER = {}
for canon, aliases in _CANON.items():
    GAZETTEER[canon.lower()] = canon
    for a in aliases:
        GAZETTEER[a] = canon


def mentioned_cities(text):
    """[(position, canonical_name)] for every gazetteer city or alias found in the text."""
    t, out = text.lower(), []
    for name, canon in GAZETTEER.items():
        m = None
        for m in re.finditer(rf"\b{re.escape(name)}\b", t):
            pass
        if m:
            out.append((m.start(), canon))
    return out