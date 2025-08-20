import os
import json
import re
import cv2
import numpy as np
import pytesseract
import requests
from functools import lru_cache
from rapidfuzz import process, fuzz
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class AnalysisResult:
    """Structured result for prescription analysis"""
    raw_text: str
    medicines: List[Dict]
    lab_tests: List[Dict]
    pairwise_notes: List[str]
    confidence_score: float
    analysis_timestamp: str
    image_hash: str

class PrescriptionAnalyzer:
    """Main class for prescription analysis with enhanced features"""
    
    def __init__(self, drug_db_path: str = "data/drug_db.json", 
                 lab_db_path: str = "data/lab_tests_db.json"):
        self.drug_db_path = drug_db_path
        self.lab_db_path = lab_db_path
        self._setup_tesseract()
        self._load_databases()
        self.analysis_cache = {}
        
    def _setup_tesseract(self):
        """Enhanced Tesseract setup with better error handling"""
        _tess_env = os.getenv("TESSERACT_CMD")
        if _tess_env and os.path.exists(_tess_env):
            pytesseract.pytesseract.tesseract_cmd = _tess_env
            return
            
        # Multiple possible locations for different OS
        possible_paths = [
            r"D:\tesseract-ocr\tesseract.exe",  # Windows
            "/usr/bin/tesseract",               # Linux
            "/opt/homebrew/bin/tesseract",      # macOS M1
            "/usr/local/bin/tesseract"          # macOS Intel
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logging.info(f"Tesseract found at: {path}")
                return
                
        logging.warning("Tesseract not found. Please install and set TESSERACT_CMD environment variable.")

    def _load_databases(self):
        """Load drug and lab databases with error handling"""
        try:
            self.drug_db = self.load_json(self.drug_db_path)
            self.lab_db = self.load_json(self.lab_db_path)
            logging.info(f"Loaded {len(self.drug_db)} drugs and {len(self.lab_db)} lab tests")
        except FileNotFoundError as e:
            logging.error(f"Database file not found: {e}")
            self.drug_db = {}
            self.lab_db = {}

    @staticmethod
    def load_json(path: str) -> Dict:
        """Load JSON with better error handling"""
        if not os.path.exists(path):
            logging.warning(f"JSON file not found: {path}. Creating empty database.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {path}: {e}")
            return {}

    def preprocess_image(self, image_path: str, enhance_contrast: bool = True) -> np.ndarray:
        """Enhanced image preprocessing with multiple techniques"""
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Could not read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR (ensure minimum resolution)
        height, width = gray.shape
        if max(height, width) < 1000:
            scale = 1000.0 / max(height, width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        # Enhance contrast if requested
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
        )
        
        return binary

    def extract_text_multi_config(self, image_path: str) -> str:
        """Extract text using multiple OCR configurations for better results"""
        img = self.preprocess_image(image_path)
        
        # Multiple OCR configurations to try
        configs = [
            "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-/|:() ",
            "--oem 3 --psm 4",
            "--oem 3 --psm 3",
            "--oem 1 --psm 6"
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                # Get text and confidence
                data = pytesseract.image_to_data(img, lang="eng", config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                text = pytesseract.image_to_string(img, lang="eng", config=config)
                text = re.sub(r"[ \t]+", " ", text).strip()
                
                if avg_confidence > best_confidence and len(text) > len(best_text) * 0.8:
                    best_text = text
                    best_confidence = avg_confidence
                    
            except Exception as e:
                logging.warning(f"OCR config failed: {config}, error: {e}")
                continue
        
        logging.info(f"Best OCR confidence: {best_confidence:.2f}")
        return best_text

    def calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash for image caching"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def tokenize_advanced(self, text: str) -> List[str]:
        """Enhanced tokenization with medical terminology handling"""
        # Preserve medical units and measurements
        text = re.sub(r"(\d+)\s*(mg|ml|mcg|g|kg|tab|cap)", r"\1\2", text, flags=re.IGNORECASE)
        
        # Clean up unwanted characters but preserve medical symbols
        text = re.sub(r"[^A-Za-z0-9\+\-/|:().\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        tokens = []
        for tok in text.split():
            # Handle compound medical terms
            parts = re.findall(r"[A-Za-z]+\d*|[0-9]+[A-Za-z]*", tok)
            if parts:
                tokens.extend(parts)
        
        return tokens

    def fuzzy_match_entities_enhanced(self, text: str, db: Dict, threshold: int = 78, 
                                    top_k: int = 10, use_context: bool = True) -> List[Tuple[str, str, int]]:
        """Enhanced entity matching with contextual scoring"""
        if not db:
            return []
            
        synonyms, canon_map = self.build_synonym_index(db)
        tokens = self.tokenize_advanced(text)
        
        # Generate candidates with different n-gram sizes
        candidates = []
        
        # Single tokens (length >= 2)
        singles = [t for t in tokens if len(t) >= 2]
        candidates.extend(singles)
        
        # Bigrams
        bigrams = [" ".join(b) for b in zip(tokens, tokens[1:])]
        candidates.extend(bigrams)
        
        # Trigrams for compound drug names
        trigrams = [" ".join(t) for t in zip(tokens, tokens[1:], tokens[2:])]
        candidates.extend(trigrams)
        
        # Add lowercase variants
        candidates.extend([c.lower() for c in candidates])
        candidates = list(set(candidates))
        
        results, seen = [], set()
        for cand in candidates:
            try:
                res = process.extractOne(cand, synonyms, scorer=fuzz.WRatio)
                if res:
                    match, score, _ = res
                    
                    # Context-based score adjustment
                    if use_context and self._has_medical_context(cand, text):
                        score = min(100, score + 5)  # Boost score for medical context
                    
                    if score >= threshold:
                        canon = canon_map[match]
                        if canon not in seen:
                            seen.add(canon)
                            results.append((canon, match, int(score)))
                            if len(results) >= top_k:
                                break
            except Exception as e:
                logging.debug(f"Error matching candidate '{cand}': {e}")
                continue
                
        return sorted(results, key=lambda x: x[2], reverse=True)

    def _has_medical_context(self, term: str, full_text: str) -> bool:
        """Check if term appears in medical context"""
        medical_indicators = [
            "tab", "tablet", "cap", "capsule", "mg", "ml", "twice", "daily", 
            "morning", "evening", "before", "after", "meals", "prescribed"
        ]
        
        term_pos = full_text.lower().find(term.lower())
        if term_pos == -1:
            return False
            
        # Check surrounding context (50 chars before and after)
        context = full_text[max(0, term_pos-50):term_pos+50].lower()
        return any(indicator in context for indicator in medical_indicators)

    def build_synonym_index(self, db: Dict) -> Tuple[List[str], Dict[str, str]]:
        """Build synonym index from database"""
        synonyms, canon_map = [], {}
        for canon, meta in db.items():
            for syn in meta.get("synonyms", []):
                s = syn.strip()
                if s:
                    synonyms.append(s)
                    canon_map[s] = canon
        return synonyms, canon_map

    def calculate_confidence_score(self, medicines: List[Dict], lab_tests: List[Dict], 
                                 raw_text: str) -> float:
        """Calculate overall confidence score for the analysis"""
        if not medicines and not lab_tests:
            return 0.0
        
        # Base confidence on OCR quality (text length and character diversity)
        text_quality = min(1.0, len(raw_text) / 500.0)  # Normalize by expected length
        char_diversity = len(set(raw_text.lower())) / 26.0  # Alphabet coverage
        
        # Average matching scores
        all_scores = []
        all_scores.extend([m["score"] for m in medicines])
        all_scores.extend([l["score"] for l in lab_tests])
        
        avg_match_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Combine factors
        confidence = (text_quality * 0.3 + char_diversity * 0.2 + avg_match_score/100 * 0.5)
        return min(1.0, confidence)

    @lru_cache(maxsize=256)
    def fetch_openfda_label_enhanced(self, generic: str, brand: str = None, 
                                   api_key: str = None) -> Dict:
        """Enhanced openFDA integration with better error handling and caching"""
        base_url = "https://api.fda.gov/drug/label.json"
        headers = {"Accept": "application/json"}
        
        # Build search queries
        queries = []
        if generic:
            queries.append(f'openfda.generic_name:"{generic.strip()}"')
        if brand:
            queries.append(f'openfda.brand_name:"{brand.strip()}"')
        
        for query in queries:
            params = {"search": query, "limit": 1}
            if api_key:
                params["api_key"] = api_key
                
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("results"):
                        result = data["results"][0]
                        return {
                            "source": "openFDA",
                            "warnings": result.get("warnings", [])[:3],
                            "drug_interactions": result.get("drug_interactions", [])[:3],
                            "boxed_warning": result.get("boxed_warning", [])[:2],
                            "dosage": result.get("dosage_and_administration", [])[:2],
                            "indications": result.get("indications_and_usage", [])[:2],
                            "contraindications": result.get("contraindications", [])[:2],
                            "adverse_reactions": result.get("adverse_reactions", [])[:2]
                        }
                elif response.status_code == 404:
                    logging.info(f"No FDA data found for: {generic or brand}")
                else:
                    logging.warning(f"FDA API returned status {response.status_code}")
                    
            except requests.RequestException as e:
                logging.error(f"Error fetching FDA data: {e}")
                
        return {}

    def check_advanced_interactions(self, canon_meds: List[str]) -> List[str]:
        """Enhanced interaction checking with more drug classes"""
        notes = []
        if not canon_meds:
            return notes
            
        # Get drug classes
        classes = {m: self.drug_db.get(m, {}).get("class") for m in canon_meds}
        
        # Check for multiple NSAIDs
        nsaids = [m for m, c in classes.items() if c == "NSAID"]
        if len(nsaids) >= 2:
            notes.append(f"⚠️ Multiple NSAIDs detected ({', '.join(nsaids)}): May increase GI bleeding risk")
        
        # Check for blood thinners + NSAIDs
        blood_thinners = [m for m, c in classes.items() if c in ["Anticoagulant", "Antiplatelet"]]
        if blood_thinners and nsaids:
            notes.append(f"⚠️ Blood thinner + NSAID combination detected: Monitor for bleeding")
        
        # Check for multiple ACE inhibitors/ARBs
        ace_arbs = [m for m, c in classes.items() if c in ["ACE Inhibitor", "ARB"]]
        if len(ace_arbs) >= 2:
            notes.append(f"⚠️ Multiple ACE inhibitors/ARBs: May cause hypotension")
        
        # Check for sedative combinations
        sedatives = [m for m, c in classes.items() if c in ["Benzodiazepine", "Sedative", "Hypnotic"]]
        if len(sedatives) >= 2:
            notes.append(f"⚠️ Multiple sedatives detected: Risk of excessive sedation")
            
        return notes

    def analyze_prescription(self, image_path: str, med_threshold: int = 82, 
                           lab_threshold: int = 75, top_k_meds: int = 6, 
                           top_k_labs: int = 10, api_key: str = None) -> AnalysisResult:
        """Main analysis function with enhanced features"""
        
        # Check cache first
        image_hash = self.calculate_image_hash(image_path)
        cache_key = f"{image_hash}_{med_threshold}_{lab_threshold}_{top_k_meds}_{top_k_labs}"
        
        if cache_key in self.analysis_cache:
            logging.info("Returning cached result")
            return self.analysis_cache[cache_key]
        
        # Extract text with enhanced OCR
        raw_text = self.extract_text_multi_config(image_path)
        
        # Match medicines and lab tests
        med_matches = self.fuzzy_match_entities_enhanced(
            raw_text, self.drug_db, threshold=med_threshold, top_k=top_k_meds
        )
        
        lab_matches = self.fuzzy_match_entities_enhanced(
            raw_text, self.lab_db, threshold=lab_threshold, top_k=top_k_labs
        )
        
        # Enrich medicine data
        medicines_detail = []
        for canon, matched, score in med_matches:
            drug_info = self.drug_db.get(canon, {})
            generic = drug_info.get("generic", "")
            
            # Fetch FDA label
            fda_label = self.fetch_openfda_label_enhanced(generic, matched, api_key)
            if not fda_label:
                fda_label = self._fallback_label_from_db(canon, drug_info)
            
            medicines_detail.append({
                "canonical": canon,
                "matched_as": matched,
                "score": score,
                "generic": generic,
                "drug_class": drug_info.get("class", "Unknown"),
                "label": fda_label
            })
        
        # Prepare lab tests
        labs_detail = [
            {
                "canonical": canon,
                "matched_as": matched,
                "score": score,
                "description": self.lab_db.get(canon, {}).get("description", "")
            }
            for canon, matched, score in lab_matches
        ]
        
        # Check interactions
        pairwise_notes = self.check_advanced_interactions([m["canonical"] for m in medicines_detail])
        
        # Calculate confidence
        confidence = self.calculate_confidence_score(medicines_detail, labs_detail, raw_text)
        
        # Create result
        result = AnalysisResult(
            raw_text=raw_text,
            medicines=medicines_detail,
            lab_tests=labs_detail,
            pairwise_notes=pairwise_notes,
            confidence_score=confidence,
            analysis_timestamp=datetime.now().isoformat(),
            image_hash=image_hash
        )
        
        # Cache result
        self.analysis_cache[cache_key] = result
        
        return result

    def _fallback_label_from_db(self, canon: str, drug_info: Dict) -> Dict:
        """Enhanced fallback with more information"""
        return {
            "source": "offline",
            "warnings": drug_info.get("common_warnings", [])[:3],
            "drug_interactions": drug_info.get("interactions", [])[:3],
            "boxed_warning": drug_info.get("boxed_warnings", [])[:2],
            "dosage": drug_info.get("dosage", [])[:2],
            "indications": drug_info.get("indications", [])[:2],
            "contraindications": drug_info.get("contraindications", [])[:2],
            "adverse_reactions": drug_info.get("side_effects", [])[:2],
            "generic": drug_info.get("generic", "")
        }

    def export_results(self, result: AnalysisResult, output_path: str):
        """Export analysis results to JSON"""
        export_data = {
            "analysis_timestamp": result.analysis_timestamp,
            "confidence_score": result.confidence_score,
            "image_hash": result.image_hash,
            "raw_text": result.raw_text,
            "medicines": result.medicines,
            "lab_tests": result.lab_tests,
            "pairwise_notes": result.pairwise_notes
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results exported to: {output_path}")

# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Prescription Analyzer')
    parser.add_argument('image_path', type=str, help='Path to prescription image')
    parser.add_argument('--drug-db', type=str, default='data/drug_db.json', help='Drug database path')
    parser.add_argument('--lab-db', type=str, default='data/lab_tests_db.json', help='Lab tests database path')
    parser.add_argument('--med-threshold', type=int, default=82, help='Medicine matching threshold')
    parser.add_argument('--lab-threshold', type=int, default=75, help='Lab test matching threshold')
    parser.add_argument('--api-key', type=str, help='OpenFDA API key')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        logging.error(f"Image file not found: {args.image_path}")
        exit(1)
    
    # Initialize analyzer
    analyzer = PrescriptionAnalyzer(args.drug_db, args.lab_db)
    
    # Analyze prescription
    result = analyzer.analyze_prescription(
        args.image_path,
        med_threshold=args.med_threshold,
        lab_threshold=args.lab_threshold,
        api_key=args.api_key
    )
    
    # Print results
    print(json.dumps({
        "confidence_score": result.confidence_score,
        "medicines_found": len(result.medicines),
        "lab_tests_found": len(result.lab_tests),
        "interaction_warnings": len(result.pairwise_notes),
        "raw_text": result.raw_text,
        "medicines": result.medicines,
        "lab_tests": result.lab_tests,
        "pairwise_notes": result.pairwise_notes
    }, indent=2, ensure_ascii=False))
    
    # Export if requested
    if args.export:
        analyzer.export_results(result, args.export)


# Add this to the end of your existing ocr_utils.py file
# Or create a new file called ocr_utils_fixed.py

# Compatibility wrapper to fix the parameter mismatch
def analyze_prescription_wrapper(image_path: str, med_threshold: int = 82, lab_threshold: int = 75, 
                               top_k_meds: int = 6, top_k_labs: int = 10, api_key: str = None) -> Dict:
    """
    Wrapper function to maintain backward compatibility with the Streamlit app
    """
    # Check if the enhanced analyzer class exists
    try:
        analyzer = PrescriptionAnalyzer()
        result = analyzer.analyze_prescription(
            image_path=image_path,
            med_threshold=med_threshold,
            lab_threshold=lab_threshold,
            top_k_meds=top_k_meds,
            top_k_labs=top_k_labs,
            api_key=api_key
        )
        
        # Convert AnalysisResult to dict for compatibility
        return {
            "raw_text": result.raw_text,
            "medicines": result.medicines,
            "lab_tests": result.lab_tests,
            "pairwise_notes": result.pairwise_notes,
            "confidence_score": result.confidence_score
        }
    except NameError:
        # Fallback to original function if PrescriptionAnalyzer doesn't exist
        return analyze_prescription(image_path, api_key=api_key)

# Override the original function name for compatibility
analyze_prescription = analyze_prescription_wrapper
