"""
Entity Extractor for Travel Queries

This module extracts countries, nationalities, and destinations from natural language text.
This is YOUR FYP contribution - not using LLM for extraction!

Example:
    "I'm Pakistani and want to go to Singapore"
    → {origin: "PAK", destination: "SGP"}
    
    "Can Americans travel visa-free to Japan?"
    → {origin: "USA", destination: "JPN"}

Features:
    - Exact matching for known country names/aliases
    - Fuzzy matching for typos (e.g., "pakisatni" → "pakistani")
"""

import re
from typing import Dict, List, Optional, Tuple


# ============================================================================
# FUZZY MATCHING (for typos)
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    if not s1 or not s2:
        return 0.0
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


# ============================================================================
# COUNTRY MAPPINGS
# ============================================================================

# ISO3 code to country name
ISO3_TO_COUNTRY = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria", "AND": "Andorra",
    "AGO": "Angola", "ATG": "Antigua and Barbuda", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan", "BHS": "Bahamas",
    "BHR": "Bahrain", "BGD": "Bangladesh", "BRB": "Barbados", "BLR": "Belarus",
    "BEL": "Belgium", "BLZ": "Belize", "BEN": "Benin", "BTN": "Bhutan",
    "BOL": "Bolivia", "BIH": "Bosnia and Herzegovina", "BWA": "Botswana", "BRA": "Brazil",
    "BRN": "Brunei", "BGR": "Bulgaria", "BFA": "Burkina Faso", "BDI": "Burundi",
    "KHM": "Cambodia", "CMR": "Cameroon", "CAN": "Canada", "CPV": "Cape Verde",
    "CAF": "Central African Republic", "TCD": "Chad", "CHL": "Chile", "CHN": "China",
    "COL": "Colombia", "COM": "Comoros", "COG": "Congo", "COD": "DR Congo",
    "CRI": "Costa Rica", "CIV": "Ivory Coast", "HRV": "Croatia", "CUB": "Cuba",
    "CYP": "Cyprus", "CZE": "Czech Republic", "DNK": "Denmark", "DJI": "Djibouti",
    "DMA": "Dominica", "DOM": "Dominican Republic", "ECU": "Ecuador", "EGY": "Egypt",
    "SLV": "El Salvador", "GNQ": "Equatorial Guinea", "ERI": "Eritrea", "EST": "Estonia",
    "SWZ": "Eswatini", "ETH": "Ethiopia", "FJI": "Fiji", "FIN": "Finland",
    "FRA": "France", "GAB": "Gabon", "GMB": "Gambia", "GEO": "Georgia",
    "DEU": "Germany", "GHA": "Ghana", "GRC": "Greece", "GRD": "Grenada",
    "GTM": "Guatemala", "GIN": "Guinea", "GNB": "Guinea-Bissau", "GUY": "Guyana",
    "HTI": "Haiti", "HND": "Honduras", "HKG": "Hong Kong", "HUN": "Hungary",
    "ISL": "Iceland", "IND": "India", "IDN": "Indonesia", "IRN": "Iran",
    "IRQ": "Iraq", "IRL": "Ireland", "ISR": "Israel", "ITA": "Italy",
    "JAM": "Jamaica", "JPN": "Japan", "JOR": "Jordan", "KAZ": "Kazakhstan",
    "KEN": "Kenya", "KIR": "Kiribati", "XKX": "Kosovo", "KWT": "Kuwait",
    "KGZ": "Kyrgyzstan", "LAO": "Laos", "LVA": "Latvia", "LBN": "Lebanon",
    "LSO": "Lesotho", "LBR": "Liberia", "LBY": "Libya", "LIE": "Liechtenstein",
    "LTU": "Lithuania", "LUX": "Luxembourg", "MAC": "Macau", "MDG": "Madagascar",
    "MWI": "Malawi", "MYS": "Malaysia", "MDV": "Maldives", "MLI": "Mali",
    "MLT": "Malta", "MHL": "Marshall Islands", "MRT": "Mauritania", "MUS": "Mauritius",
    "MEX": "Mexico", "FSM": "Micronesia", "MDA": "Moldova", "MCO": "Monaco",
    "MNG": "Mongolia", "MNE": "Montenegro", "MAR": "Morocco", "MOZ": "Mozambique",
    "MMR": "Myanmar", "NAM": "Namibia", "NRU": "Nauru", "NPL": "Nepal",
    "NLD": "Netherlands", "NZL": "New Zealand", "NIC": "Nicaragua", "NER": "Niger",
    "NGA": "Nigeria", "PRK": "North Korea", "MKD": "North Macedonia", "NOR": "Norway",
    "OMN": "Oman", "PAK": "Pakistan", "PLW": "Palau", "PSE": "Palestine",
    "PAN": "Panama", "PNG": "Papua New Guinea", "PRY": "Paraguay", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal", "QAT": "Qatar",
    "ROU": "Romania", "RUS": "Russia", "RWA": "Rwanda", "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia", "WSM": "Samoa", "SMR": "San Marino", "STP": "São Tomé and Príncipe",
    "SAU": "Saudi Arabia", "SEN": "Senegal", "SRB": "Serbia", "SYC": "Seychelles",
    "SLE": "Sierra Leone", "SGP": "Singapore", "SVK": "Slovakia", "SVN": "Slovenia",
    "SLB": "Solomon Islands", "SOM": "Somalia", "ZAF": "South Africa", "KOR": "South Korea",
    "SSD": "South Sudan", "ESP": "Spain", "LKA": "Sri Lanka", "VCT": "Saint Vincent and the Grenadines",
    "SDN": "Sudan", "SUR": "Suriname", "SWE": "Sweden", "CHE": "Switzerland",
    "SYR": "Syria", "TWN": "Taiwan", "TJK": "Tajikistan", "TZA": "Tanzania",
    "THA": "Thailand", "TLS": "Timor-Leste", "TGO": "Togo", "TON": "Tonga",
    "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TKM": "Turkmenistan", "TUV": "Tuvalu",
    "TUR": "Turkey", "UGA": "Uganda", "UKR": "Ukraine", "ARE": "United Arab Emirates",
    "GBR": "United Kingdom", "USA": "United States", "URY": "Uruguay", "UZB": "Uzbekistan",
    "VUT": "Vanuatu", "VAT": "Vatican City", "VEN": "Venezuela", "VNM": "Vietnam",
    "YEM": "Yemen", "ZMB": "Zambia", "ZWE": "Zimbabwe"
}

# Country name to ISO3 (lowercase for matching)
COUNTRY_TO_ISO3 = {name.lower(): code for code, name in ISO3_TO_COUNTRY.items()}

# Add common variations and aliases
COUNTRY_ALIASES = {
    # UK variations
    "uk": "GBR", "britain": "GBR", "england": "GBR", "british": "GBR",
    "great britain": "GBR", "united kingdom": "GBR",
    
    # US variations
    "us": "USA", "usa": "USA", "america": "USA", "american": "USA",
    "united states": "USA", "united states of america": "USA",
    
    # UAE variations
    "uae": "ARE", "dubai": "ARE", "abu dhabi": "ARE", "emirates": "ARE",
    "united arab emirates": "ARE",
    
    # Other common aliases
    "south korea": "KOR", "korea": "KOR", "korean": "KOR",
    "czech republic": "CZE", "czechia": "CZE",
    "holland": "NLD", "dutch": "NLD",
    "russia": "RUS", "russian": "RUS",
    
    # Nationality forms (demonyms) - singular and plural
    "pakistani": "PAK", "pakistanis": "PAK",
    "indian": "IND", "indians": "IND",
    "chinese": "CHN",
    "japanese": "JPN",
    "german": "DEU", "germans": "DEU",
    "french": "FRA",
    "italian": "ITA", "italians": "ITA",
    "spanish": "ESP",
    "brazilian": "BRA", "brazilians": "BRA",
    "canadian": "CAN", "canadians": "CAN",
    "australian": "AUS", "australians": "AUS",
    "mexican": "MEX", "mexicans": "MEX",
    "turkish": "TUR",
    "egyptian": "EGY", "egyptians": "EGY",
    "saudi": "SAU", "saudis": "SAU",
    "emirati": "ARE", "emiratis": "ARE",
    "singaporean": "SGP", "singaporeans": "SGP",
    "malaysian": "MYS", "malaysians": "MYS",
    "thai": "THA", "thais": "THA",
    "filipino": "PHL", "filipinos": "PHL",
    "indonesian": "IDN", "indonesians": "IDN",
    "vietnamese": "VNM",
    "bangladeshi": "BGD", "bangladeshis": "BGD",
    "sri lankan": "LKA", "sri lankans": "LKA",
    "nepali": "NPL", "nepalis": "NPL",
    "afghan": "AFG", "afghans": "AFG",
    "iranian": "IRN", "iranians": "IRN",
    "iraqi": "IRQ", "iraqis": "IRQ",
    "syrian": "SYR", "syrians": "SYR",
    "lebanese": "LBN",
    "jordanian": "JOR", "jordanians": "JOR",
    "qatari": "QAT", "qataris": "QAT",
    "kuwaiti": "KWT", "kuwaitis": "KWT",
    "omani": "OMN", "omanis": "OMN",
    "bahraini": "BHR", "bahrainis": "BHR",
    "yemeni": "YEM", "yemenis": "YEM",
    "nigerian": "NGA", "nigerians": "NGA",
    "south african": "ZAF", "south africans": "ZAF",
    "kenyan": "KEN", "kenyans": "KEN",
    "ethiopian": "ETH", "ethiopians": "ETH",
    "moroccan": "MAR", "moroccans": "MAR",
    "algerian": "DZA", "algerians": "DZA",
    "tunisian": "TUN", "tunisians": "TUN",
    "sudanese": "SDN",
    "american": "USA", "americans": "USA",
    "british": "GBR", "brits": "GBR",
}

# Merge aliases into main mapping
COUNTRY_TO_ISO3.update(COUNTRY_ALIASES)

# Origin indicators (words that suggest the following country is the origin/nationality)
ORIGIN_INDICATORS = [
    r"i'm\s+(?:a\s+)?",  # "I'm Pakistani", "I'm a Pakistani"
    r"i\s+am\s+(?:a\s+)?",  # "I am Pakistani"
    r"as\s+(?:a\s+)?",  # "as a Pakistani"
    r"being\s+(?:a\s+)?",  # "being Pakistani"
    r"from\s+",  # "from Pakistan"
    r"my\s+passport\s+is\s+",  # "my passport is Pakistani"
    r"(?:using|have|hold|with)\s+(?:a\s+)?.*?passport",  # "using a Pakistani passport"
    r"citizen\s+of\s+",  # "citizen of Pakistan"
    r"national\s+of\s+",  # "national of Pakistan"
]

# Destination indicators (words that suggest the following country is the destination)
DESTINATION_INDICATORS = [
    r"to\s+",  # "to Singapore"
    r"visit(?:ing)?\s+",  # "visiting Singapore"
    r"go(?:ing)?\s+to\s+",  # "going to Singapore"
    r"travel(?:ling|ing)?\s+to\s+",  # "travelling to Singapore"
    r"fly(?:ing)?\s+to\s+",  # "flying to Singapore"
    r"enter(?:ing)?\s+",  # "entering Singapore"
    r"into\s+",  # "into Singapore"
]

# Pattern: "X visa for Y" - X is destination, Y is origin (nationality)
# e.g., "Singapore visa for Pakistani" → destination=Singapore, origin=Pakistan
VISA_FOR_PATTERN = r"(\w+)\s+visa\s+for\s+"


class EntityExtractor:
    """
    Extracts travel-related entities from natural language text.
    
    This is YOUR implementation - not using LLM for extraction!
    
    Features:
    - Extracts origin/nationality
    - Extracts destination
    - Handles various phrasings and aliases
    - Fast (no LLM call needed)
    """
    
    def __init__(self):
        self.country_to_iso3 = COUNTRY_TO_ISO3
        self.iso3_to_country = ISO3_TO_COUNTRY
    
    def extract_countries(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract origin and destination countries from text.
        
        Args:
            text: Natural language query
            
        Returns:
            {
                'origin': 'PAK' or None,
                'origin_name': 'Pakistan' or None,
                'destination': 'SGP' or None,
                'destination_name': 'Singapore' or None,
                'all_countries': [list of all countries found]
            }
        """
        text_lower = text.lower()
        
        # Find all country mentions with their positions
        found_countries = self._find_all_countries(text_lower)
        
        # Determine origin and destination based on context
        origin, destination = self._classify_countries(text_lower, found_countries)
        
        # Check if origin was detected from a nationality form (not ambiguous)
        origin_is_nationality = False
        if origin:
            for country in found_countries:
                if country['iso3'] == origin and self._is_nationality_form(country['name']):
                    origin_is_nationality = True
                    break
        
        # Build result
        result = {
            'origin': origin,
            'origin_name': self.iso3_to_country.get(origin) if origin else None,
            'destination': destination,
            'destination_name': self.iso3_to_country.get(destination) if destination else None,
            'all_countries': [c['iso3'] for c in found_countries],
            'origin_is_nationality': origin_is_nationality,  # True = not ambiguous, definitely origin
        }
        
        return result
    
    def _find_all_countries(self, text: str) -> List[Dict]:
        """
        Find all country mentions in the text with positions.
        Uses exact matching first, then fuzzy matching for typos.
        """
        found = []
        
        # Sort country names by length (longest first) to match "United States" before "States"
        sorted_names = sorted(self.country_to_iso3.keys(), key=len, reverse=True)
        
        # Track which positions have been matched to avoid duplicates
        matched_positions = set()
        
        # PASS 1: Exact matching
        for name in sorted_names:
            # Use word boundary matching
            pattern = r'\b' + re.escape(name) + r'\b'
            for match in re.finditer(pattern, text):
                start, end = match.span()
                
                # Check if this position overlaps with already matched text
                if any(start <= pos < end for pos in matched_positions):
                    continue
                
                # Mark these positions as matched
                for pos in range(start, end):
                    matched_positions.add(pos)
                
                found.append({
                    'name': name,
                    'iso3': self.country_to_iso3[name],
                    'start': start,
                    'end': end,
                    'fuzzy': False,
                })
        
        # PASS 2: Fuzzy matching for unmatched words (typo correction)
        # Only if we didn't find anything in exact matching
        if not found:
            found = self._fuzzy_find_countries(text, matched_positions)
        
        # Sort by position
        found.sort(key=lambda x: x['start'])
        
        return found
    
    def _fuzzy_find_countries(self, text: str, already_matched: set) -> List[Dict]:
        """
        Find countries using fuzzy matching for typos.
        
        Example: "pakisatni" → "pakistani" → Pakistan
        
        Only matches words with:
        - Minimum length of 5 characters (to avoid false positives)
        - Similarity ratio >= 0.75 (75% similar)
        """
        found = []
        
        # Extract words from text
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        
        for word in words:
            # Skip if this word position was already matched
            word_start = text.lower().find(word)
            if any(word_start <= pos < word_start + len(word) for pos in already_matched):
                continue
            
            # Find best matching country name
            best_match = None
            best_ratio = 0.75  # Minimum threshold
            
            for name in self.country_to_iso3.keys():
                # Only compare with names of similar length (avoid matching "uk" with "pakistan")
                if abs(len(name) - len(word)) > 3:
                    continue
                
                ratio = similarity_ratio(word, name)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = name
            
            if best_match:
                word_start = text.lower().find(word)
                found.append({
                    'name': best_match,
                    'iso3': self.country_to_iso3[best_match],
                    'start': word_start,
                    'end': word_start + len(word),
                    'fuzzy': True,
                    'original': word,
                    'similarity': best_ratio,
                })
        
        return found
    
    def _classify_countries(self, text: str, found_countries: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify found countries as origin or destination based on context.
        
        Returns:
            (origin_iso3, destination_iso3)
        """
        if not found_countries:
            return None, None
        
        origin = None
        destination = None
        
        # Special case: "X visa for Y" pattern
        # e.g., "Singapore visa for Pakistani" → destination=Singapore, origin=Pakistan
        visa_for_match = re.search(r'(\w+)\s+visa\s+for\s+(\w+)', text)
        if visa_for_match and len(found_countries) >= 2:
            first_word = visa_for_match.group(1).lower()
            second_word = visa_for_match.group(2).lower()
            
            # Check if first word is a country (destination)
            if first_word in self.country_to_iso3:
                destination = self.country_to_iso3[first_word]
            
            # Check if second word is a nationality (origin)
            if second_word in self.country_to_iso3:
                origin = self.country_to_iso3[second_word]
            
            if origin and destination:
                return origin, destination
        
        # Special case: "for Nationality" at the end often means origin
        # e.g., "What about UAE for Pakistanis?" → Origin=Pakistan, Destination=UAE
        # BUT: "Do I need a visa for France?" - France is destination (not a nationality form)
        for_pattern_match = re.search(r'for\s+(\w+?)(s)?\s*\??$', text)
        if for_pattern_match:
            word = for_pattern_match.group(1).lower()
            has_plural_s = for_pattern_match.group(2) is not None
            word_with_s = word + 's'
            
            # Check if it's a nationality form (e.g., "Pakistani", "Americans")
            # Check both singular and plural forms
            is_nationality = self._is_nationality_form(word)
            
            if is_nationality:
                # Try to find in country mapping
                if word in self.country_to_iso3:
                    origin = self.country_to_iso3[word]
                elif word_with_s in self.country_to_iso3:
                    origin = self.country_to_iso3[word_with_s]
                
                # If we found origin from "for Nationality", the OTHER country is destination
                # e.g., "What about UAE for Pakistanis?" → UAE is destination
                if origin:
                    for country in found_countries:
                        if country['iso3'] != origin:
                            destination = country['iso3']
                            break
                    
                    if origin and destination:
                        return origin, destination
            
            # Check if the word (potentially with 's') is in country mapping as a nationality
            elif word_with_s in self.country_to_iso3 and self._is_nationality_form(word_with_s):
                origin = self.country_to_iso3[word_with_s]
                for country in found_countries:
                    if country['iso3'] != origin:
                        destination = country['iso3']
                        break
                if origin and destination:
                    return origin, destination
            
            # Otherwise it's a country name, so it's the destination
            # e.g., "visa for France" → France is destination
            elif word in self.country_to_iso3:
                destination = self.country_to_iso3[word]
        
        for country in found_countries:
            # Get text before this country mention
            text_before = text[:country['start']]
            
            # Check for origin indicators
            is_origin = False
            for pattern in ORIGIN_INDICATORS:
                if re.search(pattern + r'\s*$', text_before):
                    is_origin = True
                    break
            
            # Check for destination indicators
            is_destination = False
            for pattern in DESTINATION_INDICATORS:
                if re.search(pattern + r'\s*$', text_before):
                    is_destination = True
                    break
            
            # Assign based on indicators
            if is_origin and not origin:
                origin = country['iso3']
            elif is_destination and not destination:
                destination = country['iso3']
            elif not origin and not destination:
                # First country with no clear indicator
                # Check if it's a nationality form (likely origin)
                if self._is_nationality_form(country['name']):
                    origin = country['iso3']
        
        # If we found two countries but couldn't classify, use order
        if len(found_countries) >= 2:
            if not origin and not destination:
                # First is origin, second is destination
                origin = found_countries[0]['iso3']
                destination = found_countries[1]['iso3']
            elif origin and not destination:
                # Find a country that's not the origin
                for country in found_countries:
                    if country['iso3'] != origin:
                        destination = country['iso3']
                        break
            elif destination and not origin:
                # Find a country that's not the destination
                for country in found_countries:
                    if country['iso3'] != destination:
                        origin = country['iso3']
                        break
        
        # If only one country found, try to determine if it's origin or destination
        if len(found_countries) == 1 and not origin and not destination:
            country = found_countries[0]
            # If it's a nationality form (demonym), it's definitely the origin
            if self._is_nationality_form(country['name']):
                origin = country['iso3']
            # Otherwise, it's AMBIGUOUS - could be origin or destination
            # We return it as 'origin' but with origin_is_nationality=False
            # so ConversationState can ask for clarification
            else:
                origin = country['iso3']  # Ambiguous - let ConversationState handle
        
        return origin, destination
    
    def _is_nationality_form(self, name: str) -> bool:
        """
        Check if the word is a NATIONALITY form (e.g., 'Pakistani' vs 'Pakistan').
        
        Returns True only for words that are clearly demonyms/nationalities,
        NOT for place names that happen to end in similar letters (e.g., 'Dubai').
        """
        # These are SPECIFICALLY nationality/demonym words (people, not places)
        # Place names like "dubai", "mali", "fiji" should NOT be considered nationality forms
        KNOWN_DEMONYMS = {
            'pakistani', 'pakistanis', 'indian', 'indians', 'chinese', 'japanese',
            'german', 'germans', 'french', 'italian', 'italians', 'spanish',
            'brazilian', 'brazilians', 'canadian', 'canadians', 'australian', 'australians',
            'mexican', 'mexicans', 'american', 'americans', 'british', 'brits',
            'turkish', 'egyptian', 'egyptians', 'saudi', 'saudis',
            'emirati', 'emiratis', 'singaporean', 'singaporeans', 
            'malaysian', 'malaysians', 'thai', 'thais', 'filipino', 'filipinos',
            'indonesian', 'indonesians', 'vietnamese', 'bangladeshi', 'bangladeshis',
            'sri lankan', 'sri lankans', 'nepali', 'nepalis', 'afghan', 'afghans',
            'iranian', 'iranians', 'iraqi', 'iraqis', 'syrian', 'syrians',
            'lebanese', 'jordanian', 'jordanians', 'qatari', 'qataris',
            'kuwaiti', 'kuwatis', 'omani', 'omanis', 'bahraini', 'bahrainis',
            'yemeni', 'yemenis', 'nigerian', 'nigerians', 'south african', 'south africans',
            'kenyan', 'kenyans', 'ethiopian', 'ethiopians', 'moroccan', 'moroccans',
            'algerian', 'algerians', 'tunisian', 'tunisians', 'sudanese',
            'russian', 'russians', 'korean', 'koreans', 'dutch',
        }
        
        return name.lower() in KNOWN_DEMONYMS
    
    def get_country_name(self, iso3: str) -> Optional[str]:
        """Get country name from ISO3 code."""
        return self.iso3_to_country.get(iso3)
    
    def get_iso3(self, country_name: str) -> Optional[str]:
        """Get ISO3 code from country name."""
        return self.country_to_iso3.get(country_name.lower())


# ============================================================================
# Convenience function
# ============================================================================

def extract_countries_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    Extract origin and destination countries from text.
    
    This is a convenience function that creates an EntityExtractor instance.
    
    Args:
        text: Natural language query
        
    Returns:
        {
            'origin': 'PAK' or None,
            'origin_name': 'Pakistan' or None,
            'destination': 'SGP' or None,
            'destination_name': 'Singapore' or None,
            'all_countries': [list of all countries found]
        }
    """
    extractor = EntityExtractor()
    return extractor.extract_countries(text)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "I'm Pakistani and want to go to Singapore",
        "What visa do I need from Pakistan to UK?",
        "Can Americans travel visa-free to Japan?",
        "I am travelling to Dubai",
        "Do I need a visa for France?",
        "Pakistani passport to Singapore",
        "As a British citizen, can I visit Canada?",
        "I'm from India and going to Thailand",
        "What about UAE for Pakistanis?",
        "Singapore visa for Pakistani",
    ]
    
    extractor = EntityExtractor()
    
    print("=" * 70)
    print("ENTITY EXTRACTOR TEST")
    print("=" * 70)
    
    for query in test_queries:
        result = extractor.extract_countries(query)
        print(f"\nQuery: {query}")
        print(f"  Origin: {result['origin_name']} ({result['origin']})")
        print(f"  Destination: {result['destination_name']} ({result['destination']})")

