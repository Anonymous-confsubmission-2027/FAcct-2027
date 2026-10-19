"""Data preparation for FAMOUS.

Builds the labelled FakeNewsNet table, assigns prominence (Tranco top 10,000) and
credibility (reliability labels) to each publisher domain, draws the 150-article
sample, and writes the curated domain pools from which the stand-in sources are drawn.

Two exclusion lists are used, in the order in which they were made:
  data/excluded_sources_articles.csv  80 sources (unidentifiable, platforms, reference,
      official sites...), created on 5 October 2026 and applied to the ARTICLES before
      the sample was drawn (seed 30, 8 October 2026);
  data/excluded_domains.csv           the final list of 293 domains, which extends the
      first one (11 October 2026) and is applied to the DOMAIN POOLS only.
At the end, the script compares the sample and the pools it produced with the files
used in the experiments (data/articles_150.csv and data/domain_pools.csv).

The draw depends on the order in which os.listdir returns the article folders, which
can differ between file systems; the sample was drawn on macOS. If the comparison
fails on another system, data/articles_150.csv remains the reference sample.

Inputs (not redistributed): FakeNewsNet_Dataset/ (the four crawled folders, downloaded
on 6 September 2026), data/top-1m_notoriete.csv (Tranco list K9QYW), data/credibilite.csv
(news-media reliability labels of 28 October 2024).
"""
import os
import json
import csv
from urllib.parse import urlparse
from collections import Counter
import random
import re  


FICHIER_ENTREE = "data/fakenewsnet_complet_v5.csv"
DOSSIER_SORTIE = "data_out"
FICHIER_EXCLUSIONS = "data/excluded_sources_articles.csv"      # first list (80 sources), applied to the articles
FICHIER_EXCLUSIONS_DOMAINES = "data/excluded_domains.csv"      # final list (293 domains), applied to the domain pools
FICHIER_ECHANTILLON_REFERENCE = "data/articles_150.csv"        # the sample used in all experiments
FICHIER_POOLS_REFERENCE = "data/domain_pools.csv"              # the pools used to draw the stand-in sources


def charger_categorie(dossier_racine, label, nom_dossier):
    lignes = []
    for nom_article in os.listdir(dossier_racine):
        chemin_json = os.path.join(dossier_racine, nom_article, 'news_article.json')
        if os.path.exists(chemin_json):
            with open(chemin_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['id'] = nom_article
                data['label'] = label
                # Cleanup: keep only the first element of the authors list (often the real name)
                if isinstance(data.get('authors'), list) and len(data['authors']) > 0:
                    data['authors'] = data['authors'][0]
                else:
                    data['authors'] = None
                # Keep only the useful fields (the rest, like meta_data, is too complex for a simple CSV)
                data_utile = {
                    'id': data.get('id'),
                    'label': data.get('label'),
                    'dossier_origine': nom_dossier,  # new column: gossipcop_fake, gossipcop_real, politifact_fake or politifact_real
                    'title': data.get('title'),
                    'text': data.get('text'),
                    'source': data.get('source'),
                    'authors': data.get('authors'),
                    'url': data.get('url'),
                    'publish_date': data.get('publish_date'),
                }
                lignes.append(data_utile)
    return lignes


def ecrire_csv(chemin, lignes):
    with open(chemin, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=lignes[0].keys())
        writer.writeheader()
        writer.writerows(lignes)

def lire_csv(chemin):
    with open(chemin, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
    
def creer_csv_complet():
    # Load the 4 strata from FakeNewsNet_Dataset
    toutes_les_lignes = []
    toutes_les_lignes += charger_categorie('FakeNewsNet_Dataset/gossipcop_fake', 'fake', 'gossipcop_fake')
    toutes_les_lignes += charger_categorie('FakeNewsNet_Dataset/gossipcop_real', 'real', 'gossipcop_real')
    toutes_les_lignes += charger_categorie('FakeNewsNet_Dataset/politifact_fake', 'fake', 'politifact_fake')
    toutes_les_lignes += charger_categorie('FakeNewsNet_Dataset/politifact_real', 'real', 'politifact_real')

    print("Total number of articles:", len(toutes_les_lignes))

    # Collect all possible columns
    toutes_les_colonnes = set()
    for ligne in toutes_les_lignes:
        toutes_les_colonnes.update(ligne.keys())
    toutes_les_colonnes = list(toutes_les_colonnes)

    # Write the CSV
    with open('data/fakenewsnet_complet.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=toutes_les_colonnes)
        writer.writeheader()
        writer.writerows(toutes_les_lignes)

    print("Step 1 completed: Creation of the complete data file: fakenewsnet_complet.csv")


def remplir_champs_vides():
    lignes = []
    with open('data/fakenewsnet_complet.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for ligne in reader:
            if not ligne['source']:  # empty field
                ligne['source'] = 'inconnu'
            if not ligne['authors']:
                ligne['authors'] = 'inconnu'
            lignes.append(ligne)

    # Save the new CSV
    ecrire_csv('data/fakenewsnet_complet_v2.csv', lignes)

    print('Step 2 completed: empty fields were filled with "inconnu": fakenewsnet_complet_v2.csv')


def formater_nom_media():
    lignes = []
    with open('data/fakenewsnet_complet_v2.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for ligne in reader:
            url = ligne['source']
            if url == 'inconnu':
                domaine = 'inconnu'
            else:
                if not url.startswith('http'):
                    url = 'http://' + url
                domaine = urlparse(url).netloc.replace('www.', '')
            ligne['domaine'] = domaine
            lignes.append(ligne)

    # Save the new CSV
    ecrire_csv('data/fakenewsnet_complet_v3.csv', lignes)

    print("Step 3 completed: Creation of the 'domaine' column with formatted domain names: fakenewsnet_complet_v3.csv")


def chercher_notoriete(domaine, classement):
    if domaine == "inconnu":
        return "s1_peu_connue", domaine

    # remove spaces and convert to lowercase
    domaine = domaine.lower().strip()

    # Only one dot in the domain: simple domain
    if domaine.count(".") == 1:
        if domaine in classement:
            return "s0_connue", domaine
        else:
            return "s1_peu_connue", domaine

    # several dots in the domain: so it is a subdomain
    # check whether the subdomains are in the ranking, by combining the parts of the domain
    else:
        parties = domaine.split(".")

        for i in range(len(parties) - 1):
            domaine_test = ".".join(parties[i:])

            if domaine_test in classement:
                return "s0_connue", domaine_test
        return "s1_peu_connue", domaine


def attribuer_notoriete():      
    classement = {}
    with open("data/top-1m_notoriete.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, (rang, domaine) in enumerate(reader):
            if i >= 10000: 
                break
            classement[domaine] = int(rang)

    print(f"{len(classement)} domaines chargés (top 10 000 seulement)") 

    with open("data/fakenewsnet_complet_v3.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lignes = list(reader)

    for l in lignes:
        notoriete, domaine_principal = chercher_notoriete(l["domaine"], classement)

        l["notoriete"] = notoriete
        l["domaine_principal"] = domaine_principal

    # Save the new CSV
    ecrire_csv('data/fakenewsnet_complet_v4.csv', lignes)
    print("Step 4 completed: notoriety and main domain assigned: fakenewsnet_complet_v4.csv")


def donner_credibilite(domaine, credibilite_dict):
    label = credibilite_dict.get(domaine)
    if label == "1":
        return "credible"
    elif label == "0":
        return "mixte"
    elif label == "-1":
        return "peu_credible"
    else:
        return "non_note"
    

def attribuer_credibilite():
    # load the credibility list into a dictionary
    credibilite_dict = {}
    with open("data/credibilite.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for ligne in reader:
            credibilite_dict[ligne["domain"]] = ligne["reliability_label"]

    print(f"{len(credibilite_dict)} media loaded from the credibility list")

    # load the fakenewsnet_complet_v4.csv dataset
    with open("data/fakenewsnet_complet_v4.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lignes = list(reader)

    for l in lignes:
        l["credibilite"] = donner_credibilite(l["domaine_principal"], credibilite_dict)

    # Save the new CSV
    ecrire_csv('data/fakenewsnet_complet_v5.csv', lignes)
    print("Step 5 completed: credibility attribution: fakenewsnet_complet_v5.csv")


def charger_exclusions(chemin):
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        return {r["domaine_principal"] for r in csv.DictReader(f)}


def source_de(l):
    # returns whether the article comes from gossipcop or politifact, based on the article id
    return "gossipcop" if l["id"].startswith("gossipcop") else "politifact"


## returns True if the article title is the name of the domain or of the main domain
def titre_est_la_source(l):
    titre = l["title"].strip().lower()
    return titre == l["domaine"].lower() or titre == l["domaine_principal"].lower()


# function that replaces the domain or domaine_principal name with the neutral word "media" in the text "texte"
def remplacer_domaine(texte, l):
    for nom in (l["domaine"], l["domaine_principal"]):
        texte = texte.replace(nom, "media")
    return texte


def nettoyer_titre(l):
    titre = l["title"].strip()
    sep = r"[\-\|:–—]"

    for nom in (l["domaine"], l["domaine_principal"]):
        n = re.escape(nom)
        titre = re.sub(rf"^\s*{n}\s*{sep}?\s*", "", titre, flags=re.IGNORECASE)   # at the beginning: remove the media name
        titre = re.sub(rf"\s*{sep}?\s*{n}\s*$", "", titre, flags=re.IGNORECASE)   # at the end: remove the media name
        titre = re.sub(rf"\s*{n}\s*", " media ", titre, flags=re.IGNORECASE)      # in the middle: replace with the neutral word "media"

    return re.sub(r"\s+", " ", titre).strip()


def echantillonner_n(lignes, n):
    strates = sorted({(source_de(l), l["label"]) for l in lignes})
    nbr_article_cellule = n // len(strates)
    reste = n % len(strates) # the number of articles left to draw after taking an equal number from each stratum

    random.shuffle(strates)
    resultat = []

    for i, (s, lab) in enumerate(strates):

        k = nbr_article_cellule + (1 if i < reste else 0)
        cellule = [l for l in lignes if source_de(l) == s and l["label"] == lab]
        k = min(k, len(cellule)) # don't exceed the number of articles allocated to the cell
        print(f"  {s} / {lab} : {len(cellule)} disponibles, on tire {k}")
        resultat += random.sample(cellule, k)

    # if articles are missing, fill in with random remaining ones
    if len(resultat) < n:
        print(f"Missing {n - len(resultat)} article(s), completing...")
        articles_pris = {l["id"] for l in resultat}
        restants = [l for l in lignes if l["id"] not in articles_pris]
        resultat += random.sample(restants, n - len(resultat))

    random.shuffle(resultat)
    return resultat


def resumer_categories(lignes, titre):
    # displays for each category A, B, C, D: number of sources, number of articles, proportion of fake and real
    categories = {
        "A": ("s0_connue", "credible"),
        "B": ("s0_connue", "peu_credible"),
        "C": ("s1_peu_connue", "credible"),
        "D": ("s1_peu_connue", "peu_credible"),
    }
    print(f"--- {titre} ---")
    print("cat | sources | articles | fake | real")

    for nom, (notoriete, credibilite) in categories.items():
        cat_articles = [l for l in lignes if l["notoriete"] == notoriete and l["credibilite"] == credibilite]
        sources = {l["domaine_principal"] for l in cat_articles}
        nbr_fake = sum(1 for l in cat_articles if l["label"] == "fake")
        nbr_real = sum(1 for l in cat_articles if l["label"] == "real")
        proportion_fake = 100 * nbr_fake / len(cat_articles) if cat_articles else 0
        proportion_real = 100 * nbr_real/ len(cat_articles) if cat_articles else 0
        
        print(f" {nom}  | {len(sources):7} | {len(cat_articles):8} | {proportion_fake:5.1f} % | {proportion_real:5.1f} %")
    print()


def retirer_doublons(lignes):
    articles_uniques = {}
    for l in lignes:
        cle = (l["url"], l["title"], l["text"])
        if cle not in articles_uniques:
            articles_uniques[cle] = l 
    return list(articles_uniques.values())


def creer_echantillon():
    random.seed(30)
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    lignes = lire_csv(FICHIER_ENTREE)
    nbr_lignes = len(lignes)
    print(f"Total articles : {nbr_lignes}")

    print("removing duplicate articles (same url, same title, same text)")
    lignes = retirer_doublons(lignes)
    print(f"duplicates removed : {nbr_lignes - len(lignes)}")

    print("starting to exclude article entries whose domains are in the exclusion list :")
    resumer_categories(lignes, "BEFORE exclusion")

    exclus = charger_exclusions(FICHIER_EXCLUSIONS)
    
    lignes = [l for l in lignes if l["domaine_principal"] not in exclus 
                and l["title"].strip() 
                and l["text"].strip() 
                and not titre_est_la_source(l)]
        
    resumer_categories(lignes, "AFTER exclusion")

    # un domaine = one row (domaine_principal, notoriete, credibilite)
    domaines = {}
    for l in lignes:
        d = l["domaine_principal"]
        if d != "inconnu":
            domaines[d] = {"domaine_principal": d, "notoriete": l["notoriete"], "credibilite": l["credibilite"]}

    # the final exclusion list applies to the domain pools only (not to the articles)
    domaines_exclus = charger_exclusions(FICHIER_EXCLUSIONS_DOMAINES)
    domaines = {d: domaines[d] for d in domaines if d not in domaines_exclus}


    ecrire_csv(os.path.join(DOSSIER_SORTIE, "domaines_valides.csv"), [domaines[d] for d in sorted(domaines)])

    print("Starting the sampling of 150 articles")
    echantillon = echantillonner_n(lignes, 150)
    
    for l in echantillon:
        l["title_original"] = l["title"] 
        l["text_original"] = l["text"]
        l["title"] = nettoyer_titre(l)  # overwrite title with the cleaned version
        l["text"] = remplacer_domaine(l["text"], l) # overwrite text with the cleaned version
    
    print(Counter((source_de(l), l["label"]) for l in echantillon))
    ecrire_csv(os.path.join(DOSSIER_SORTIE, "echantillon_150.csv"), echantillon)


def separer_domaines_par_categorie():

    liste = lire_csv(os.path.join(DOSSIER_SORTIE, "domaines_valides.csv"))
    colonnes = ["domaine_principal", "notoriete", "credibilite"]

    categories = {
        "A": ("s0_connue",     "credible"),
        "B": ("s0_connue",     "peu_credible"),
        "C": ("s1_peu_connue", "credible"),
        "D": ("s1_peu_connue", "peu_credible"),
    }

    dossier = os.path.join(DOSSIER_SORTIE, "categorie_domaine")
    os.makedirs(dossier, exist_ok=True) # create the folder if it doesn't exist

    for nom, (notoriete, credibilite) in categories.items():
        selection = [d for d in liste if d["notoriete"] == notoriete and d["credibilite"] == credibilite]

        with open(os.path.join(dossier, f"domaines_{nom}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=colonnes)
            writer.writeheader()
            writer.writerows(selection)

        print(f"Category {nom} : {len(selection)} domains")


def comparer_echantillon():
    # Compares the sample just drawn with the sample used in the experiments:
    # same articles, in the same order, with the same labels, titles and texts.
    nouveau = lire_csv(os.path.join(DOSSIER_SORTIE, "echantillon_150.csv"))
    reference = lire_csv(FICHIER_ECHANTILLON_REFERENCE)
    colonnes = ("id", "label", "domaine_principal", "title", "text", "title_original", "text_original")
    identique = len(nouveau) == len(reference) and all(
        n[c] == r[c] for n, r in zip(nouveau, reference) for c in colonnes)
    if identique:
        print("Sample: identical to data/articles_150.csv (same 150 articles, same order, same texts).")
    else:
        communs = len({n["id"] for n in nouveau} & {r["id"] for r in reference})
        print(f"Sample: DIFFERENT from data/articles_150.csv ({communs} of 150 articles in common).")
        print("  The draw depends on the order in which the article folders are listed, which can")
        print("  differ between file systems. data/articles_150.csv remains the reference sample.")
    return identique


def comparer_pools():
    # Compares the curated domain pools with those used to draw the stand-in sources.
    colonnes = ("domaine_principal", "notoriete", "credibilite")
    nouveau = sorted(tuple(d[c] for c in colonnes) for d in lire_csv(os.path.join(DOSSIER_SORTIE, "domaines_valides.csv")))
    reference = sorted(tuple(d[c] for c in colonnes) for d in lire_csv(FICHIER_POOLS_REFERENCE))
    identique = nouveau == reference
    print(f"Domain pools: {'identical to' if identique else 'DIFFERENT from'} data/domain_pools.csv "
          f"({len(nouveau)} domains, reference {len(reference)}).")
    return identique


def signaler_editeurs_exclus():
    # Lists the sampled articles whose publisher is excluded as a stand-in by the final list.
    # They stay in the sample: the final list governs which domains may serve as stand-in
    # sources, and the original publisher is never shown to the models.
    exclus = charger_exclusions(FICHIER_EXCLUSIONS_DOMAINES)
    echantillon = lire_csv(FICHIER_ECHANTILLON_REFERENCE)
    hors = sorted((l["domaine_principal"], l["id"]) for l in echantillon if l["domaine_principal"] in exclus)
    print(f"{len(hors)} sampled articles have a publisher excluded as a stand-in by the final list:")
    for d, i in hors:
        print(f"  {i:24} {d}")


def main():
    print("Pipeline start")
    # convert my folders "FakeNewsNet_Dataset/" into a single CSV file
    creer_csv_complet()
    # fill empty source and author fields with "inconnu"
    remplir_champs_vides()
    # create another column that would contain the formatted media name without https and www
    formater_nom_media()
    # assign the notoriety attribute to each media in our dataset, based on the tranco list
    attribuer_notoriete()
    # assign the credibility attribute to each media in our dataset, based on the credibility list
    attribuer_credibilite()
    # step 6: create the sample of 150 articles (first exclusion list, duplicates removed, seed 30)
    # and the curated domain pools (final exclusion list)
    creer_echantillon()
    # step 7: create the 4 files for the 4 categories A, B, C, D
    separer_domaines_par_categorie()
    # step 8: compare with the files used in the experiments
    comparer_echantillon()
    comparer_pools()
    signaler_editeurs_exclus()
    print("Pipeline end")


if __name__ == "__main__":
    main()