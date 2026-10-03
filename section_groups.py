"""Mutually exclusive topic groups for normalized level-2 Wikipedia headings.

First matching group wins. Subsections inherit their level-2 parent; this is
an article-structure classification, not semantic classification of sentences.
"""

from extract_links import norm_heading

SECTION_GROUPS = [
    ("Personalities", ("personnalit",)),
    ("Toponymy", ("toponym", "nom des habitants", "gentile")),
    ("History", ("histoire", "historique", "prehistoire", "antiquite", "moyen age", "epoque")),
    ("Culture & heritage", ("culture", "patrimoine", "monument", "lieux", "tourisme", "heraldique",
                            "vie locale", "sport", "cultes", "festiv", "gastronom", "arts")),
    ("Economy", ("economie", "activite", "emploi", "agricult", "industrie", "commerce", "entreprise")),
    ("Politics & administration", ("politique", "administr", "jumelage", "budget", "fiscal", "intercommun",
                                   "liste des maires", "tendances politiques")),
    ("Population & society", ("population", "demograph", "enseignement", "sante", "equipement", "service",
                              "societe", "logement")),
    ("Urbanism", ("urbanism", "occupation des sols", "morphologie", "risques", "voies", "transport", "acces")),
    ("Geography", ("geograph", "localisation", "situation", "hydrograph", "climat", "relief", "geologie",
                   "milieux naturels", "communes limitrophes")),
]


def section_group(sec: str) -> str:
    sec = norm_heading(str(sec))
    if sec == "lead":
        return "Lead paragraph"
    for name, keys in SECTION_GROUPS:
        if any(k in sec for k in keys):
            return name
    return "Other"
