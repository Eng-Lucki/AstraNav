"""
Yale Bright Star Catalogue (BSC5) loader.

The BSC5 lists every star brighter than about visual magnitude 6.5 (the
naked-eye limit), 9110 entries, with J2000 positions. File used:
data/bsc5_catalog.gz, the original fixed-width "catalog" file from the
CDS/VizieR archive, catalogue V/50 (Hoffleit & Warren 1991).
"""
import gzip
import os
from collections import Counter

import numpy as np

from attitude import radec_to_vec

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CATALOG_PATH = os.path.join(DATA_DIR, "bsc5_catalog.gz")
CATALOG_URL = "https://vizier.cfa.harvard.edu/ftp/cats/V/50/catalog.gz"

GREEK = {"Alp": "α", "Bet": "β", "Gam": "γ", "Del": "δ", "Eps": "ε", "Zet": "ζ",
         "Eta": "η", "The": "θ", "Iot": "ι", "Kap": "κ", "Lam": "λ", "Mu": "μ",
         "Nu": "ν", "Xi": "ξ", "Omi": "ο", "Pi": "π", "Rho": "ρ", "Sig": "σ",
         "Tau": "τ", "Ups": "υ", "Phi": "φ", "Chi": "χ", "Psi": "ψ", "Ome": "ω"}

CONSTELLATION_NAMES = {
    "And": "Andromeda", "Ant": "Antlia", "Aps": "Apus", "Aqr": "Aquarius", "Aql": "Aquila", "Ara": "Ara",
    "Ari": "Aries", "Aur": "Auriga", "Boo": "Bootes", "Cae": "Caelum", "Cam": "Camelopardalis",
    "Cnc": "Cancer", "CVn": "Canes Venatici", "CMa": "Canis Major", "CMi": "Canis Minor",
    "Cap": "Capricornus", "Car": "Carina", "Cas": "Cassiopeia", "Cen": "Centaurus", "Cep": "Cepheus",
    "Cet": "Cetus", "Cha": "Chamaeleon", "Cir": "Circinus", "Col": "Columba", "Com": "Coma Berenices",
    "CrA": "Corona Australis", "CrB": "Corona Borealis", "Crv": "Corvus", "Crt": "Crater", "Cru": "Crux",
    "Cyg": "Cygnus", "Del": "Delphinus", "Dor": "Dorado", "Dra": "Draco", "Equ": "Equuleus",
    "Eri": "Eridanus", "For": "Fornax", "Gem": "Gemini", "Gru": "Grus", "Her": "Hercules",
    "Hor": "Horologium", "Hya": "Hydra", "Hyi": "Hydrus", "Ind": "Indus", "Lac": "Lacerta", "Leo": "Leo",
    "LMi": "Leo Minor", "Lep": "Lepus", "Lib": "Libra", "Lup": "Lupus", "Lyn": "Lynx", "Lyr": "Lyra",
    "Men": "Mensa", "Mic": "Microscopium", "Mon": "Monoceros", "Mus": "Musca", "Nor": "Norma",
    "Oct": "Octans", "Oph": "Ophiuchus", "Ori": "Orion", "Pav": "Pavo", "Peg": "Pegasus", "Per": "Perseus",
    "Phe": "Phoenix", "Pic": "Pictor", "Psc": "Pisces", "PsA": "Piscis Austrinus", "Pup": "Puppis",
    "Pyx": "Pyxis", "Ret": "Reticulum", "Sge": "Sagitta", "Sgr": "Sagittarius", "Sco": "Scorpius",
    "Scl": "Sculptor", "Sct": "Scutum", "Ser": "Serpens", "Sex": "Sextans", "Tau": "Taurus",
    "Tel": "Telescopium", "Tri": "Triangulum", "TrA": "Triangulum Australe", "Tuc": "Tucana",
    "UMa": "Ursa Major", "UMi": "Ursa Minor", "Vel": "Vela", "Vir": "Virgo", "Vol": "Volans",
    "Vul": "Vulpecula",
}

# Traditional names of bright stars: HR number -> (name, expected Bayer designation).
# The designation is checked against the catalogue on load, so a typo can never mislabel a star.
PROPER_NAMES = {
    15: ("Alpheratz", "Alp And"), 21: ("Caph", "Bet Cas"), 168: ("Schedar", "Alp Cas"),
    264: ("Navi", "Gam Cas"), 337: ("Mirach", "Bet And"), 403: ("Ruchbah", "Del Cas"),
    424: ("Polaris", "Alp UMi"), 472: ("Achernar", "Alp Eri"), 542: ("Segin", "Eps Cas"),
    603: ("Almach", "Gam And"), 617: ("Hamal", "Alp Ari"), 936: ("Algol", "Bet Per"),
    1017: ("Mirfak", "Alp Per"), 1165: ("Alcyone", "Eta Tau"), 1457: ("Aldebaran", "Alp Tau"),
    1708: ("Capella", "Alp Aur"), 1713: ("Rigel", "Bet Ori"), 1790: ("Bellatrix", "Gam Ori"),
    1791: ("Elnath", "Bet Tau"), 1852: ("Mintaka", "Del Ori"), 1903: ("Alnilam", "Eps Ori"),
    1948: ("Alnitak", "Zet Ori"), 2004: ("Saiph", "Kap Ori"), 2061: ("Betelgeuse", "Alp Ori"),
    2088: ("Menkalinan", "Bet Aur"), 2326: ("Canopus", "Alp Car"), 2421: ("Alhena", "Gam Gem"),
    2491: ("Sirius", "Alp CMa"), 2618: ("Adhara", "Eps CMa"), 2693: ("Wezen", "Del CMa"),
    2891: ("Castor", "Alp Gem"), 2943: ("Procyon", "Alp CMi"), 2990: ("Pollux", "Bet Gem"),
    3982: ("Regulus", "Alp Leo"), 4295: ("Merak", "Bet UMa"), 4301: ("Dubhe", "Alp UMa"),
    4534: ("Denebola", "Bet Leo"), 4554: ("Phecda", "Gam UMa"), 4660: ("Megrez", "Del UMa"),
    4730: ("Acrux", "Alp Cru"), 4853: ("Mimosa", "Bet Cru"), 4905: ("Alioth", "Eps UMa"),
    5054: ("Mizar", "Zet UMa"), 5056: ("Spica", "Alp Vir"), 5191: ("Alkaid", "Eta UMa"),
    5267: ("Hadar", "Bet Cen"), 5340: ("Arcturus", "Alp Boo"), 5459: ("Rigil Kentaurus", "Alp Cen"),
    5563: ("Kochab", "Bet UMi"), 5793: ("Alphecca", "Alp CrB"), 6134: ("Antares", "Alp Sco"),
    6527: ("Shaula", "Lam Sco"), 6556: ("Rasalhague", "Alp Oph"), 6705: ("Eltanin", "Gam Dra"),
    7001: ("Vega", "Alp Lyr"), 7417: ("Albireo", "Bet Cyg"), 7528: ("Fawaris", "Del Cyg"),
    7557: ("Altair", "Alp Aql"), 7790: ("Peacock", "Alp Pav"), 7796: ("Sadr", "Gam Cyg"),
    7924: ("Deneb", "Alp Cyg"), 7949: ("Aljanah", "Eps Cyg"), 8308: ("Enif", "Eps Peg"),
    8425: ("Alnair", "Alp Gru"), 8728: ("Fomalhaut", "Alp PsA"), 8781: ("Markab", "Alp Peg"),
}


class StarCatalog:
    def __init__(self, path=CATALOG_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Star catalogue not found at {path}.\nDownload it from {CATALOG_URL} "
                "and save it there (keep the .gz, it is read compressed).")
        hr, ra, dec, mag, con, bayer, flam = [], [], [], [], [], [], []
        with gzip.open(path, "rt", encoding="latin-1") as fh:
            for line in fh:
                # Byte positions from the catalogue ReadMe (1-based 76-90: J2000 RA/Dec, 103-107: V mag)
                if len(line) < 107 or not line[75:77].strip() or not line[102:107].strip():
                    continue   # a few entries are novae / non-stellar objects without positions
                ra_deg = 15.0 * (int(line[75:77]) + int(line[77:79]) / 60 + float(line[79:83]) / 3600)
                dec_deg = int(line[84:86]) + int(line[86:88]) / 60 + int(line[88:90]) / 3600
                if line[83] == "-":
                    dec_deg = -dec_deg
                hr.append(int(line[0:4]))
                ra.append(ra_deg)
                dec.append(dec_deg)
                mag.append(float(line[102:107]))
                flam.append(line[4:7].strip())
                bayer.append(line[7:10].strip())
                con.append(line[11:14].strip())

        self.hr = np.array(hr)
        self.ra = np.array(ra)
        self.dec = np.array(dec)
        self.mag = np.array(mag)
        self.con = np.array(con)
        self.bayer = np.array(bayer)
        self.flamsteed = np.array(flam)
        self.vec = radec_to_vec(self.ra, self.dec)
        self._row = {h: i for i, h in enumerate(self.hr)}

        self.proper = {}
        for h, (name, desig) in PROPER_NAMES.items():
            i = self._row.get(h)
            if i is not None and f"{self.bayer[i]} {self.con[i]}" == desig:
                self.proper[i] = name

    def __len__(self):
        return len(self.hr)

    def label(self, i, greek=False):
        """Best human-readable name: proper name, Bayer letter, Flamsteed number, or HR number."""
        if i in self.proper:
            return self.proper[i]
        if self.bayer[i]:
            letter = GREEK.get(self.bayer[i], self.bayer[i]) if greek else self.bayer[i]
            return f"{letter} {self.con[i]}"
        if self.flamsteed[i]:
            return f"{self.flamsteed[i]} {self.con[i]}"
        return f"HR {self.hr[i]}"

    def dominant_constellation(self, indices):
        cons = [self.con[i] for i in indices if self.con[i]]
        if not cons:
            return ""
        abbr = Counter(cons).most_common(1)[0][0]
        return CONSTELLATION_NAMES.get(abbr, abbr)


_catalog = None


def get_catalog():
    """Load the catalogue once and share it."""
    global _catalog
    if _catalog is None:
        _catalog = StarCatalog()
    return _catalog
