# app.py - Interface Streamlit pour le projet Adult Census
# Lancer avec :  streamlit run app.py   (dans le même dossier que modele_adult.pkl)
import joblib
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Prédiction du revenu - Adult", page_icon="💼", layout="wide")

BINS = [0, 30, 45, 60, 100]
LABELS = ['Jeune', 'Adulte', 'Senior', 'Aine']


@st.cache_resource
def charger():
    return joblib.load("modele_adult.pkl")


try:
    art = charger()
except FileNotFoundError:
    st.error("Fichier modele_adult.pkl introuvable : exécute d'abord la Cellule 22 du notebook "
             "et lance l'application depuis le même dossier.")
    st.stop()


def preparer(saisie):
    """Même prétraitement que le notebook : outliers -> age_group -> OneHot -> normalisation."""
    num = pd.DataFrame([{c: float(saisie[c]) for c in art['num_cols']}])
    for c, (bas, haut) in art['bornes'].items():          # mêmes bornes que le train
        num[c] = num[c].clip(bas, haut)

    age_label = pd.cut(num['Age'], bins=BINS, labels=LABELS).astype(str)
    num_norm = pd.DataFrame(art['scaler'].transform(num[art['num_cols']]),
                            columns=art['num_cols'])      # transform seulement

    ligne = pd.DataFrame(0.0, index=[0], columns=art['colonnes'])
    for c in art['num_cols']:
        ligne[c] = num_norm[c].values
    ligne['age_group'] = art['le_age'].transform(age_label)
    for c in art['cat_cols']:                             # OneHot manuel (drop_first : la
        cle = f"{c}_{saisie[c]}"                          # modalité de référence = tout à 0)
        if cle in ligne.columns:
            ligne[cle] = 1.0
    return ligne, num_norm.values


def verifier(saisie):
    """Contrôle de saisie : avertissements (incohérences) et infos (valeurs ramenées aux bornes)."""
    avert, infos = [], []
    rel, sexe = saisie['Relationship'].strip(), saisie['Sex'].strip()
    if rel == 'Husband' and sexe == 'Female':
        avert.append("« Husband » avec le sexe Female : combinaison incohérente.")
    if rel == 'Wife' and sexe == 'Male':
        avert.append("« Wife » avec le sexe Male : combinaison incohérente.")
    if saisie['Age'] < 22 and saisie['Education_Num'] >= 13:
        avert.append("Moins de 22 ans avec un Bachelors ou plus : combinaison très rare.")
    if saisie['Capital_Gain'] > 0 and saisie['Capital_Loss'] > 0:
        avert.append("Gain ET perte en capital > 0 : cas quasi inexistant dans les données.")
    noms = {'Capital_Gain': 'Gain en capital', 'Capital_Loss': 'Perte en capital',
            'Hours_per_week': 'Heures par semaine'}
    for c, (bas, haut) in art['bornes'].items():
        if not bas <= saisie[c] <= haut:
            infos.append(f"{noms[c]} = {saisie[c]:g} sera ramené à [{bas:g} ; {haut:g}] "
                         f"(bornes du train, traitement des outliers).")
    return avert, infos


def choix(nom, options, defaut=None):
    i = options.index(defaut) if defaut in options else 0
    return st.selectbox(nom, options, index=i, format_func=str.strip)


st.title("💼 Prédiction du revenu - Adult Census")
tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Performances des modèles"])

with tab1:
    noms_modeles = list(art['modeles'])
    nom = st.selectbox("Modèle", noms_modeles, key='modele_choisi',
                       index=noms_modeles.index(art['nom_defaut']),
                       help="Les 3 modèles ont été optimisés par GridSearch. "
                            "KNN permet en plus de voir les profils les plus proches.")
    mod = art['modalites']
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Âge", 17, 90, 38)
        education = choix("Niveau d'éducation",
                          sorted(mod['Education'], key=lambda e: art['edu_map'][e]), ' Bachelors')
        heures = st.slider("Heures de travail par semaine", 1, 99, 40)
        gain = st.number_input("Gain en capital", 0, 99999, 0)
        perte = st.number_input("Perte en capital", 0, 4356, 0)
        sexe = choix("Sexe", mod['Sex'])
    with c2:
        workclass = choix("Type d'emploi", mod['Workclass'], ' Private')
        statut = choix("Situation maritale", mod['Martial_Status'], ' Married-civ-spouse')
        metier = choix("Profession", mod['Occupation'], ' Prof-specialty')
        relation = choix("Relation familiale", mod['Relationship'], ' Husband')
        race = choix("Origine (Race)", mod['Race'], ' White')
        pays = choix("Pays", mod['Country'], ' United-States')
        reel = st.selectbox("Revenu réel (facultatif, pour vérifier la prédiction)",
                            ["Inconnu", "<=50K", ">50K"], key='reel')

    saisie = {'Age': age, 'Education_Num': art['edu_map'][education],
              'Capital_Gain': gain, 'Capital_Loss': perte, 'Hours_per_week': heures,
              'Workclass': workclass, 'Education': education, 'Martial_Status': statut,
              'Occupation': metier, 'Relationship': relation, 'Race': race,
              'Sex': sexe, 'Country': pays}

    avert, infos = verifier(saisie)
    for a in avert:
        st.warning(a)
    for i in infos:
        st.info(i)

    if st.button("Prédire", type="primary"):
        st.session_state['predit'] = True

    if st.session_state.get('predit'):
        ligne, num_norm = preparer(saisie)
        le = art['le_cible']

        def predire(m):
            classe = le.inverse_transform(m.predict(ligne))[0]
            proba = m.predict_proba(ligne)[0][list(m.classes_).index(1)]
            return classe, proba

        modele = art['modeles'][nom]
        classe, proba = predire(modele)

        st.subheader(f"Résultat ({nom})")
        if proba >= 0.5:
            st.success(f"Revenu prédit : **{classe}**")
        else:
            st.info(f"Revenu prédit : **{classe}**")
        st.metric("Probabilité de gagner plus de 50K", f"{proba:.1%}")
        st.progress(float(proba))

        if reel != "Inconnu":
            if reel == classe:
                st.success(f"✅ Le modèle a trouvé la bonne classe ({reel}).")
            else:
                st.error(f"❌ Le modèle s'est trompé : réel = {reel}, prédit = {classe}.")
            st.caption("Un seul exemple ne mesure pas la qualité du modèle : "
                       "voir l'onglet Performances (évaluation sur tout adult_test).")

        # Explication KNN : les voisins qui ont voté
        if isinstance(modele, KNeighborsClassifier):
            dist, idx = modele.kneighbors(ligne)
            vois = art['voisins_info'].iloc[idx[0]].copy()
            vois['Target'] = vois['Target'].str.strip()
            vois.insert(0, 'Distance', dist[0].round(2))
            n_sup = int((vois['Target'] == '>50K').sum())
            vote = " (vote pondéré par la distance)" if modele.weights == 'distance' else ""
            st.markdown(f"**Pourquoi cette réponse ?** Parmi les {len(vois)} profils "
                        f"d'entraînement les plus proches, **{n_sup}** gagnent >50K et "
                        f"**{len(vois) - n_sup}** gagnent <=50K{vote}.")
            st.dataframe(vois.reset_index(drop=True))

        st.markdown("**Avis des 3 modèles :**")
        lignes = []
        for n, m in art['modeles'].items():
            c, p = predire(m)
            lignes.append({'Modèle': n, 'Prédiction': c, 'Probabilité >50K': f"{p:.1%}"})
        st.dataframe(pd.DataFrame(lignes).set_index('Modèle'))

        cluster = int(art['kmeans'].predict(num_norm)[0])
        st.markdown(f"**Profil (K-Means) : cluster {cluster}** - profil moyen de ce cluster :")
        st.dataframe(art['profils'].loc[[cluster]])
        st.caption("Estimation statistique issue d'un jeu de données historique : "
                   "elle ne constitue pas une décision.")

with tab2:
    st.subheader("Modèles optimisés (évalués sur adult_test)")
    st.dataframe(art['df_gs'])
    st.subheader("Comparaison avec les modèles de référence (paramètres par défaut)")
    st.dataframe(art['df_metriques'])
    st.subheader("Variables les plus importantes (Random Forest)")
    st.bar_chart(art['importances_var'].head(10))
