# FraudGuard — Document de Conception & Spécification Complète
# Projet MLOps — Détection de Fraude Bancaire
# Mastère Spécialisé IA — Télécom Paris

<p align="center">
  <img src="architecture.png" alt="Architecture système FraudGuard" width="900">
</p>

---

## 1. Contexte du projet

### 1.1 Le problème métier
La fraude bancaire coûte des milliards d'euros par an aux institutions financières. L'objectif est de détecter en temps réel les transactions frauduleuses parmi un volume massif de transactions légitimes, en appliquant les principes MLOps pour garantir un système fiable, traçable et maintenable en production.

### 1.2 Le dataset
- **Source** : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Volume** : 284 807 transactions sur 2 jours (septembre 2013)
- **Features** : V1-V28 (résultats d'une ACP = Analyse en Composantes Principales, anonymisés pour confidentialité), Time (secondes depuis la 1ère transaction), Amount (montant en euros)
- **Target** : Class (0 = transaction légitime, 1 = fraude)
- **Déséquilibre** : seulement 492 fraudes sur 284 807 transactions → **0.172% de fraudes**. Un modèle qui prédit "légitime" à chaque fois aurait 99.83% d'accuracy mais serait complètement inutile.

### 1.3 Sources de code
- **Notebook Kaggle** : "Anomaly Detection LightGBM Isolation Forest" (georgeyoussef1) — inspiration pour la logique ML
- **Repo GitHub** : https://github.com/Ahmedfekhfakh/Fraudguard — socle existant initié par Ahmed

### 1.4 Philosophie
Les profs ont été clairs : la performance du modèle n'est pas le sujet. Ce qui compte, c'est **tout ce qu'il y a autour** — orchestration, tracking, déploiement, monitoring, CI/CD, continuous training (= réentraînement automatique). Le code ML du notebook = ~5% de la valeur. Le système MLOps autour = ~95%.

---

## 1.5 Bilan : État d'avancement

| # | Objectif | Statut | Détails |
|---|----------|--------|---------|
| 1 | Extraire et prétraiter les données | ✅ Fait | `ingest_and_preprocess` dans le DAG Airflow |
| 2 | Construire un modèle ML | ✅ Fait | LightGBM + Isolation Forest |
| 3 | Model Registry MLflow | ✅ Fait | Promotion Production/Staging automatique |
| 4 | Pipeline Airflow | ✅ Fait | DAG `fraud_detection_pipeline` |
| 5 | Tracking MLflow | ✅ Fait | Params, métriques, artefacts, matrice de confusion |
| 6 | API FastAPI | ✅ Fait | /predict, /predict_batch, /health, /model_metrics |
| 7 | WebApp Streamlit | ✅ Fait | Dashboard, prédiction, batch, métriques modèle |
| 8 | Continuous Training (CT) | ✅ Fait | DAG `fraud_retraining_ct` (@daily) |
| 9 | Dockerisation | ✅ Fait | 9 services Docker Compose |
| 10 | Versionning GitHub | ✅ Fait | Repo structuré, README, CLAUDE.md |
| — | Tests pytest | ✅ Fait | 4 fichiers de tests + conftest.py |
| — | Makefile | ✅ Fait | Targets test, lint, format, up, down |
| — | LocalStack S3 (artefacts) | ✅ Fait | Remplace volume Docker local |
| — | pgAdmin | ✅ Fait | Administration PostgreSQL (port 5051) |
| 9p2 | Déploiement Kubernetes | ❌ À faire | Aucun manifest `k8s/` |
| 9p3 | CI/CD GitHub Actions | ❌ À faire | Aucun workflow `.github/workflows/` |
| 12 | Monitoring Prometheus/Grafana | ❌ À faire | Pas d'endpoint `/metrics`, pas de service prometheus/grafana |
| 11 | 🌟 Données en live | ❌ À faire | Bonus |
| 13 | 🌟 Tests de charge (Locust) | ✅ Fait | Service Locust (port 8089) + 3 scénarios |
| 14 | 🌟 Rollback modèles | ❌ À faire | Bonus |
| 15 | 🌟 Human in the loop | ❌ À faire | Bonus |
| 16 | 🌟 Gestion accès API | ❌ À faire | Bonus |
| 17 | 🌟 Alerting par mail | ❌ À faire | Bonus |

---

## 2. Audit du repo existant

### 2.1 Arborescence actuelle
```
Fraudguard/
├── airflow/
│   ├── dags/
│   │   ├── fraud_pipeline.py            ← Pipeline principal (ingestion → entraînement → enregistrement)
│   │   └── fraud_retraining_ct.py       ← DAG CT (@daily) : check perf + drift → trigger réentraînement
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── uv.lock
├── api/
│   ├── main.py                          ← API FastAPI (/predict, /predict_batch, /health, /model_metrics)
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── uv.lock
├── mlflow/
│   └── Dockerfile                       ← MLflow (deps pré-installées au build)
├── webapp/                              ← Interface Streamlit complète
│   ├── app.py                           ← Point d'entrée multi-pages
│   ├── Dockerfile
│   ├── pyproject.toml / uv.lock
│   ├── config.py
│   ├── api/                             ← Client HTTP vers l'API FastAPI
│   │   ├── client.py
│   │   └── models.py
│   ├── pages/                           ← Pages Streamlit (dashboard, prédiction, batch, métriques)
│   │   ├── dashboard.py
│   │   ├── single_prediction.py
│   │   ├── batch_analysis.py
│   │   └── model_metrics.py
│   ├── components/                      ← Composants réutilisables (header, sidebar, charts, formulaire)
│   ├── styles/                          ← Thème CSS
│   └── assets/                          ← Logos
├── load_tests/
│   └── locustfile.py                  ← Tests de charge Locust (3 scénarios)
├── tests/                               ← Tests pytest
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_pipeline.py
├── conception/
│   ├── conception.md                    ← Ce document
│   ├── architecture.drawio
│   └── architecture.png
├── artifacts/                           ← Artefacts locaux (gitignored)
├── docker-compose.yml                   ← 9 services Docker
├── init-db.sql                          ← Script SQL pour la base MLflow
├── Makefile                             ← Raccourcis : make test, lint, format, up, down
├── pyproject.toml                       ← Configuration ruff
├── .env / .env.example                  ← Variables d'environnement (AWS, ports, credentials)
├── .gitignore
├── CLAUDE.md
└── README.md
```

### 2.2 Services Docker Compose

Docker Compose lance **9 services** (= 9 conteneurs) qui communiquent entre eux :

**`postgres`** (postgres:14-alpine) — port 5432 (interne, pas exposé)
Base de données partagée : stocke les métadonnées d'Airflow (état des DAGs) et de MLflow (expériences, métriques).

**`mlflow`** (Dockerfile dédié : `mlflow/Dockerfile`) — port 5000
Serveur MLflow : interface web pour voir les expériences + registre de modèles. Les dépendances sont pré-installées au build via uv. Artifacts stockés sur LocalStack S3.

**`airflow`** (apache/airflow:2.8.1) — port 8080
Orchestrateur : exécute les pipelines (DAGs) automatiquement. Interface web pour les surveiller. Login : admin/admin.

**`api`** (python:3.11-slim) — port 8000
API FastAPI : reçoit une transaction en JSON, la passe au modèle, retourne "fraude" ou "légitime" avec une probabilité.

**`webapp`** (Streamlit) — port 8501
Interface graphique pour interagir avec l'API : dashboard, prédiction unitaire, analyse batch, métriques du modèle.

**`pgadmin`** — port 5051
Interface web d'administration pour PostgreSQL. Permet de visualiser les bases Airflow et MLflow.

**`localstack`** — port 4566
Émulateur AWS local. Fournit un service S3 compatible pour stocker les artefacts MLflow (modèles, scaler, etc.) sans compte AWS réel.

**`localstack-init`** (conteneur éphémère)
Initialise le bucket S3 sur LocalStack au démarrage, puis s'arrête.

**`locust`** (locustio/locust:latest) — port 8089
Tests de montée en charge. Interface web pour simuler des utilisateurs concurrents sur l'API. 3 scénarios : transactions normales (×3), transactions suspectes (×1), health check (×1). Wait time : 0.5-2s entre requêtes.

### 2.3 DAG existant : `fraud_detection_pipeline`

Un DAG (Directed Acyclic Graph) dans Airflow, c'est une suite de tâches qui s'exécutent dans un ordre précis. Celui-ci fait :

```
ingest_and_preprocess → [train_isolation_forest, train_lightgbm] → register_best_model
```

- **Fréquence : `schedule=None`** → ce DAG ne se lance pas automatiquement. Il doit être déclenché manuellement depuis l'interface Airflow ou par un autre DAG. C'est voulu : on ne veut pas réentraîner les modèles en boucle sans raison. C'est le DAG CT (voir section 3.4) qui décide quand il faut le relancer.

**Tâche 1 : `ingest_and_preprocess`** (préparation des données)
- Charge le fichier CSV des transactions
- Supprime la colonne Time (pas utile pour la prédiction)
- Normalise la colonne Amount (pour que les montants de 1€ et 10 000€ soient comparables)
- Sépare en jeu d'entraînement (80%) et jeu de test (20%), en gardant le même ratio de fraudes dans les deux (= split stratifié)
- Sauvegarde le tout en fichiers Parquet (format optimisé)

**Tâche 2a : `train_isolation_forest`** (entraînement modèle non-supervisé)
- Entraîne un Isolation Forest (voir section 3.1 pour les explications)
- Enregistre dans MLflow : les paramètres utilisés, les métriques de performance, la matrice de confusion (tableau qui montre les erreurs)

**Tâche 2b : `train_lightgbm`** (entraînement modèle supervisé)
- Entraîne un LightGBM (voir section 3.1 pour les explications)
- Enregistre dans MLflow : paramètres, métriques, matrice de confusion, importance des features (quelles colonnes influencent le plus la prédiction)

**Tâche 3 : `register_best_model`** (choisir le meilleur modèle)
- Compare les deux modèles sur la métrique **AUC-PR** (voir section 3.2 pour l'explication)
- Le meilleur est **promu** en "Production" dans MLflow (voir section 3.3 pour ce que "promouvoir" veut dire)
- Le moins bon reste en "Staging" (= en attente, prêt à prendre le relais si besoin)
- Écrit le nom du gagnant dans un fichier `best_model.txt` que l'API lit au démarrage

### 2.4 API existante : `api/main.py`

L'API est le point d'accès pour les utilisateurs et applications qui veulent savoir si une transaction est frauduleuse :

- `GET /` : informations générales sur le projet
- `GET /health` : "est-ce que l'API fonctionne et a bien chargé un modèle ?"
- `POST /predict` : envoie une transaction (29 chiffres : V1 à V28 + Amount) → reçoit une réponse du type `{"is_fraud": false, "fraud_probability": 0.003, "risk_level": "LOW"}`
- `POST /predict_batch` : même chose mais pour un lot de transactions (max 1000 à la fois)
- `GET /model_metrics` : retourne les métriques de performance du modèle actuellement en production (récupérées depuis MLflow)

### 2.5 Problèmes identifiés

1. **~~`.env` manquant~~** ✅ — Résolu. Le fichier `.env.example` existe et `.env` est dans `.gitignore`.
2. **CSV monté depuis `../creditcard.csv`** — Le chemin suppose que le fichier CSV est dans un dossier au-dessus du projet. Fragile, ne marchera que sur la machine d'Ahmed.
3. **L'API dépend de `best_model.txt`** — Ce fichier est créé par Airflow. Si on lance l'API sans avoir d'abord exécuté le DAG Airflow, elle démarre sans modèle (retourne 503 sur `/predict`).
4. **~~Aucun test~~** ✅ — Résolu. 4 fichiers de tests créés dans `tests/` (test_api.py, test_preprocessing.py, test_model.py, test_pipeline.py) + conftest.py.
5. **~~Pas de WebApp~~** ✅ — Résolu. Application Streamlit complète dans `webapp/` avec pages, composants, API client, thème CSS et assets.
6. **~~Pas de DAG de Continuous Training~~** ✅ — Résolu. Le DAG `fraud_retraining_ct` (@daily) vérifie la performance et le drift, et déclenche le réentraînement si nécessaire.
7. **Pas de Kubernetes / CI/CD** — Aucun fichier de déploiement K8s, aucun workflow GitHub Actions.
8. **Pas de monitoring** — L'API n'expose pas de métriques pour Prometheus, pas de dashboard Grafana.
9. **~~Artefacts en volume Docker local~~** ✅ — Résolu. Les artefacts MLflow sont maintenant stockés sur **LocalStack S3** (émulateur AWS local), accessible depuis tous les services. Le bucket est initialisé automatiquement par le service `localstack-init`.
10. **~~MLflow installait ses deps au démarrage~~** ✅ — Résolu. `mlflow/Dockerfile` pré-installe les dépendances au build, `start_period` du healthcheck réduit de 120s à 10s.

### 2.6 Tests de charge Locust

Le service Locust (`load_tests/locustfile.py`) permet de tester la montée en charge de l'API FastAPI. Interface web accessible sur http://localhost:8089.

**3 scénarios avec pondération :**
- `predict_normal` (poids ×3) : envoie une transaction normale sur `/predict`
- `predict_fraud` (poids ×1) : envoie une transaction suspecte sur `/predict`
- `health_check` (poids ×1) : vérifie la disponibilité via `/health`

**Données de test :**
- `NORMAL_TRANSACTION` : valeurs V1-V28 + Amount typiques d'une transaction légitime
- `FRAUD_TRANSACTION` : valeurs V1-V28 + Amount typiques d'une transaction frauduleuse

**Configuration :** wait_time entre 0.5s et 2s entre chaque requête par utilisateur simulé. Le service dépend de l'API (healthcheck).

---

## 3. Concepts clés expliqués

### 3.1 Les deux modèles — qui fait quoi ?

On entraîne deux modèles, mais ils n'ont **pas le même rôle** en production :

#### LightGBM — le modèle qui tourne en production

LightGBM est un modèle **supervisé** : pendant l'entraînement, il a vu des exemples de transactions **avec leur étiquette** ("cette transaction était une fraude", "celle-ci était légitime"). Il a appris à reconnaître les patterns.

En production, quand on lui envoie une nouvelle transaction, il retourne une **probabilité de fraude** (ex: 0.003 = 0.3% de chance que ce soit une fraude, ou 0.87 = 87% de chance). C'est ce modèle qui est chargé dans l'API FastAPI et qui répond aux requêtes `/predict`.

Il gagnera quasi toujours la comparaison car un modèle supervisé (qui a vu les réponses pendant l'entraînement) bat presque toujours un modèle non-supervisé (qui n'a pas vu les réponses).

#### Isolation Forest — un modèle non-supervisé avec deux utilités

L'Isolation Forest est un modèle **non-supervisé** : pendant l'entraînement, il n'a **pas** regardé la colonne "fraude/légitime". Il a simplement appris à quoi ressemblent des transactions "normales" en regardant la distribution des données. Quand il voit une transaction qui ne ressemble pas aux autres, il la considère comme une "anomalie".

Ce modèle a **deux utilités** dans notre projet :

**Utilité 1 : Filet de sécurité dans le pipeline de training (déjà en place)**

Le DAG `fraud_detection_pipeline` entraîne les deux modèles puis compare leurs performances. En temps normal, le LightGBM gagne haut la main. Mais si un jour il y a un problème (bug dans le code, données corrompues, mauvais hyperparamètres), l'Isolation Forest est là comme **plan B**. Si l'IF obtient un meilleur score que le LightGBM, c'est lui qui sera mis en production automatiquement. C'est un filet de sécurité.

**Utilité 2 : Sentinelle pour détecter le "data drift" dans le DAG CT (✅ créé)**

Le "data drift" (dérive des données), c'est quand les nouvelles données ne ressemblent plus à celles sur lesquelles le modèle a été entraîné. Par exemple : le profil des fraudeurs change, les montants moyens augmentent, de nouveaux patterns apparaissent. Quand ça arrive, le modèle en production fait des prédictions de plus en plus mauvaises **sans qu'on le sache** (car en production réelle, on n'a pas les vrais labels immédiatement).

L'Isolation Forest peut servir de **sentinelle** pour détecter ce problème :

1. On prend un lot de nouvelles transactions (ex: les 1000 dernières)
2. On les passe dans l'Isolation Forest (qui a appris ce qui est "normal" à l'entraînement)
3. L'IF nous dit quel pourcentage de ces transactions lui semblent "bizarres" (= anomalies)
4. En temps normal, ce pourcentage devrait être autour de 0.2% (le taux de fraude du dataset)
5. **Si ce pourcentage monte brusquement** (ex: 8% des transactions sont "bizarres") → ça veut dire que les nouvelles données ne ressemblent plus à celles de l'entraînement
6. C'est un signal d'alerte : il faut **réentraîner le modèle** sur des données plus récentes

Pourquoi c'est malin ? Parce qu'on détecte le problème **sans attendre les vrais labels** (qui arrivent parfois des semaines après en vraie vie).

```
Résumé visuel :

LightGBM ──────────► Chargé dans l'API ────► Répond aux requêtes /predict
                     (c'est le "titulaire")    "Cette transaction a 87% de chance d'être une fraude"

Isolation Forest ──► DAG de training ────────► Comparé au LightGBM sur l'AUC-PR
                     (rôle : filet de          En temps normal il perd. Mais s'il gagne,
                      sécurité)                il prend la place du LightGBM automatiquement.

                 ──► DAG de Continuous ──────► Analyse les nouvelles transactions en batch :
                     Training                  "Habituellement 0.2% d'anomalies, aujourd'hui 8%
                     (rôle : sentinelle)        → ALERTE ! Les données ont changé.
                                                → Déclenche un réentraînement du LightGBM."
```

### 3.2 AUC-PR — la métrique de comparaison

AUC-PR = **Area Under the Precision-Recall Curve** (Aire sous la courbe Précision-Rappel).

Pour comprendre, il faut d'abord comprendre les deux composantes :
- **Precision** (Précision) : parmi toutes les transactions que le modèle a étiquetées "fraude", combien étaient réellement des fraudes ? → "Quand il dit fraude, a-t-il raison ?"
- **Recall** (Rappel) : parmi toutes les vraies fraudes du dataset, combien le modèle en a-t-il trouvé ? → "Trouve-t-il toutes les fraudes ?"

La courbe Precision-Recall trace comment ces deux métriques évoluent quand on change le seuil de décision (à partir de quelle probabilité on dit "fraude"). L'AUC-PR est l'aire sous cette courbe : plus elle est proche de 1, mieux c'est.

**Pourquoi l'AUC-PR et pas la ROC-AUC classique ?**

Quand les classes sont très déséquilibrées (0.17% de fraudes), la ROC-AUC est trompeuse : un modèle médiocre peut afficher une ROC-AUC de 0.95 juste parce qu'il est bon pour dire "légitime" (ce qui est facile vu qu'il y a 99.83% de légitimes). L'AUC-PR est beaucoup plus sévère : elle ne récompense que les modèles qui trouvent réellement les fraudes.

```
ROC-AUC  → "Distingue-t-il les deux classes en général ?"
            Trompeur quand une classe domine (99.83% vs 0.17%)

AUC-PR   → "Trouve-t-il les fraudes sans trop de fausses alertes ?"
            Fiable même avec un déséquilibre extrême
            C'est CELLE QU'ON UTILISE dans le projet
```

Dans le code, c'est la fonction `average_precision_score` de scikit-learn.

### 3.3 Champion / Challenger et "Promote" — le système de mise en production

MLflow Model Registry (le registre de modèles) fonctionne comme un système de promotion :

**Champion** = le modèle actuellement en production, celui que l'API utilise pour répondre aux requêtes. Dans MLflow, son stage (étape) est **"Production"**.

**Challenger** = un modèle candidat qui vient d'être entraîné et qu'on compare au champion. Dans MLflow, son stage est **"Staging"** (= en attente de validation).

**Promote** (promouvoir) = faire passer un modèle de Staging à Production. Concrètement, c'est un appel à la fonction MLflow `transition_model_version_stage(stage="Production")`. Après cette opération, l'API chargera le nouveau modèle au prochain redémarrage.

**Le garde-fou** : on ne promeut un challenger que s'il a une meilleure AUC-PR que le champion actuel. Si le nouveau modèle est moins bon, il reste en Staging et le champion garde sa place. C'est ce qui empêche de dégrader la production par accident.

```
┌─────────────────┐                          ┌──────────────────┐
│    Staging       │     promote              │   Production     │
│  (candidat en    │ ──────────────────────►  │  (modèle actif   │
│   attente)       │  seulement si            │   dans l'API)    │
│                  │  AUC-PR nouveau          │                  │
│  Nouveau modèle  │  > AUC-PR ancien         │  Chargé par      │
│  fraîchement     │                          │  FastAPI au      │
│  entraîné        │  sinon : on ne fait      │  démarrage       │
│                  │  rien, l'ancien reste    │                  │
└─────────────────┘                          └──────────────────┘
```

### 3.4 Continuous Training (CT) — le réentraînement automatique

Le Continuous Training est le principe de réentraîner automatiquement le modèle quand c'est nécessaire, sans intervention humaine. C'est un concept central du MLOps (vu en cours : spécifique aux systèmes ML, n'existe pas en DevOps classique).

Dans notre projet, le CT est implémenté par le **DAG `fraud_retraining_ct`** (✅ créé dans `airflow/dags/fraud_retraining_ct.py`). Ce DAG ne réentraîne pas lui-même : il **vérifie** si un réentraînement est nécessaire, et si oui, il **déclenche** le DAG de training existant (`fraud_detection_pipeline`). Le CT est le "déclencheur intelligent", le training pipeline est "l'exécutant".

### 3.5 Fréquences de déclenchement des DAGs

**`fraud_detection_pipeline`** → Fréquence : **`schedule=None`** (= jamais automatiquement)
Ce DAG est lourd (entraîne 2 modèles, ~5 min). On ne veut pas le lancer en boucle. Il est déclenché **manuellement** la première fois (pour créer le modèle initial) puis **par le DAG CT** quand un réentraînement est nécessaire.

**`fraud_retraining_ct`** → Fréquence : **`@daily`** (= une fois par jour)
Le DAG CT est léger (il vérifie juste les métriques et le drift, pas d'entraînement). Un check quotidien est un bon compromis : assez fréquent pour détecter les problèmes rapidement, pas trop pour éviter de surcharger le système. En production réelle, on pourrait le mettre à `@hourly` si le volume de transactions est élevé. Il peut aussi être déclenché **manuellement** depuis l'interface Airflow.

```
Cycle de vie des DAGs :

                      ┌────────────────────────────────────────┐
                      │      fraud_retraining_ct               │
                      │      Fréquence : @daily (tous les jours)│
Jour 1 ──────────────►│  check performance → OK                │
                      │  check drift → OK                      │
                      │  → Résultat : RAS, ne rien faire       │
                      └────────────────────────────────────────┘

                      ┌────────────────────────────────────────┐
                      │      fraud_retraining_ct               │
Jour 2 ──────────────►│  check performance → OK                │
                      │  check drift → ALERTE ! 8% d'anomalies│
                      │  → Résultat : DRIFT DÉTECTÉ            │
                      │  → Déclenche fraud_detection_pipeline  │───────►  Réentraînement
                      └────────────────────────────────────────┘          + comparaison
                                                                          + promote si meilleur
```

### 3.6 Prometheus et /metrics — le monitoring de l'API

Prometheus est un système de surveillance qui fonctionne comme un médecin de garde :
- Toutes les 15 secondes, il interroge l'API sur l'endpoint `GET /metrics`
- L'API répond avec des compteurs en texte brut (combien de requêtes, quelle latence, combien de fraudes détectées)
- Prometheus stocke ces chiffres avec un horodatage
- **Grafana** se branche sur Prometheus pour dessiner des graphiques en temps réel

L'endpoint `/metrics` n'existe pas encore dans le code d'Ahmed — c'est à ajouter.

---

## 4. Architecture cible

```
Flux de données global :

creditcard.csv
      │
      ▼
  Airflow ──────► S3 (LocalStack)    MLflow
  (DAG 1)        parquet + scaler    (registry)
      │                │                 │
      └── train ──────►└── modèle ──────►│
                                         │
                        Production model │
                              │          │
                              ▼          │
  Utilisateur ──► Webapp ──► API ◄───────┘
                  :8501      :8000
                              ▲
                              │
                           Locust
                           :8089
```

### 4.1 Services Docker Compose (environnement de développement)

```
┌───────────────────────────────────────────────────────────────────┐
│              docker-compose.yml  —  9 services                    │
│                                                                   │
│  ┌────────────┐           ┌─────────────┐                         │
│  │  postgres   │◄──────────│   pgadmin    │                        │
│  │   :5432     │           │   :5051      │                        │
│  └─────┬──────┘           └─────────────┘                         │
│        │                                                          │
│        │  ┌─────────────┐   ┌─────────────────┐                   │
│        │  │ localstack   │──►│ localstack-init  │                   │
│        │  │   :4566      │   │ (crée bucket S3) │                   │
│        │  └──────┬──────┘   └────────┬────────┘                   │
│        │         │                   │                             │
│        ▼         ▼                   ▼                             │
│  ┌──────────────────────────────────────┐                         │
│  │              mlflow                   │                         │
│  │              :5000                    │                         │
│  │  metadata: postgres │ artifacts: S3   │                         │
│  └───────┬──────────────────┬───────────┘                         │
│          │                  │                                      │
│    ┌─────┘                  └──────┐                               │
│    ▼                               ▼                               │
│ ┌──────────┐                ┌──────────┐                          │
│ │ airflow   │                │   api     │                          │
│ │  :8080    │                │  :8000    │                          │
│ └──────────┘                └────┬─────┘                          │
│                                  │                                 │
│                            ┌─────┴──────┐                          │
│                            ▼            ▼                          │
│                      ┌──────────┐ ┌──────────┐                    │
│                      │  webapp   │ │  locust   │                    │
│                      │  :8501    │ │  :8089    │                    │
│                      └──────────┘ └──────────┘                    │
│                                                                   │
│  ❌ À FAIRE : prometheus (:9090) + grafana (:3000)                │
└───────────────────────────────────────────────────────────────────┘
```

MLflow est configuré pour stocker les artefacts sur LocalStack S3 (émulateur AWS local). Le bucket est créé automatiquement par le service `localstack-init` au démarrage. Cela permet aux modèles de persister indépendamment des conteneurs et d'être accessibles depuis tous les services.

### 4.2 DAGs Airflow (avec fréquences)

```
DAG 1 : fraud_detection_pipeline  (schedule=None, max_active_runs=1)

  ┌───────────────────────────┐
  │  ingest_and_preprocess     │
  │  CSV → parquet + scaler    │
  └────────────┬──────────────┘
               │
         ┌─────┴──────┐
         ▼            ▼
  ┌────────────┐ ┌────────────┐
  │ train_      │ │ train_      │    ← en parallèle
  │ isolation_  │ │ lightgbm    │
  │ forest      │ │             │
  └──────┬─────┘ └──────┬─────┘
         └───────┬──────┘
                 ▼
  ┌───────────────────────────┐
  │  register_best_model       │
  │  compare AUC-PR → promote  │
  │  winner → Production       │
  └───────────────────────────┘
```

```
DAG 2 : fraud_retraining_ct  (@daily, max_active_runs=1)

  ┌──────────────────┐  ┌──────────────────┐
  │ check_model_      │  │ check_data_       │    ← en parallèle
  │ performance       │  │ drift             │
  │ (AUC-PR < 0.7 ?) │  │ (anomalies > 5×?) │
  └────────┬─────────┘  └────────┬─────────┘
           └───────┬─────────────┘
                   ▼
       ┌───────────────────────┐
       │   decide_retraining    │  (BranchPythonOperator)
       └─────┬────────────┬────┘
             ▼            ▼
     ┌──────────────┐ ┌──────────────┐
     │ trigger_      │ │ skip_         │
     │ retraining    │ │ retraining    │
     │ (→ DAG 1)    │ │ (log OK)      │
     └──────────────┘ └──────────────┘
```

### 4.3 Cluster Kubernetes (environnement de production) — ❌ À FAIRE

```
Cluster Kubernetes (Docker Desktop — images locales, pas de registry)
│
├── namespace: serving
│   ├── Deployment: fraud-api (2 replicas, imagePullPolicy: Never)
│   ├── Service NodePort: :30800 → :8000  (accessible via http://localhost:30800)
│   ├── Deployment: fraud-webapp (1 replica, imagePullPolicy: Never)
│   └── Service NodePort: :30850 → :8501  (accessible via http://localhost:30850)
│
└── (bonus) namespace: monitoring
    ├── Deployment: prometheus
    ├── Deployment: grafana
    └── Service NodePort: :30300 → :3000  (accessible via http://localhost:30300)
```

`imagePullPolicy: Never` = Kubernetes utilise les images Docker construites localement, pas de registry distant. Ça marche car Docker Desktop partage le même daemon entre le host et K8s.

### 4.4 Pipeline CI/CD (GitHub Actions) — ❌ À FAIRE

CI = Continuous Integration (vérification automatique du code à chaque push)
CD = Continuous Deployment (déploiement automatique quand le code est validé)

```
            push / PR                    merge → main
                │                              │
                ▼                              ▼
         ┌────────────┐                ┌────────────────┐
         │     CI      │                │       CD        │
         │             │                │                 │
         │ ruff check  │                │ docker build    │
         │ pytest      │                │ kubectl apply   │
         └────────────┘                └────────────────┘
                                        (images locales,
                                         imagePullPolicy:
                                         Never)
```

### 4.5 Arborescence cible — état d'avancement

```
Fraudguard/
├── .github/
│   └── workflows/
│       ├── ci.yml                          ← ❌ À FAIRE
│       └── cd.yml                          ← ❌ À FAIRE
├── airflow/
│   ├── dags/
│   │   ├── fraud_pipeline.py               ← ✅ FAIT
│   │   └── fraud_retraining_ct.py          ← ✅ FAIT
│   ├── Dockerfile                          ← ✅ FAIT
│   ├── pyproject.toml                      ← ✅ FAIT
│   └── uv.lock                             ← ✅ FAIT
├── api/
│   ├── main.py                             ← ✅ FAIT (❌ manque /metrics Prometheus)
│   ├── Dockerfile                          ← ✅ FAIT
│   ├── pyproject.toml                      ← ✅ FAIT (❌ manque prometheus_client)
│   └── uv.lock                             ← ✅ FAIT
├── mlflow/
│   └── Dockerfile                          ← ✅ FAIT
├── webapp/                                 ← ✅ FAIT (complet)
│   ├── app.py                              ← ✅ FAIT
│   ├── Dockerfile                          ← ✅ FAIT
│   ├── pyproject.toml / uv.lock            ← ✅ FAIT
│   ├── config.py                           ← ✅ FAIT
│   ├── api/ (client.py, models.py)         ← ✅ FAIT
│   ├── pages/ (dashboard, prédiction, batch, métriques) ← ✅ FAIT
│   ├── components/ (header, sidebar, charts, formulaire) ← ✅ FAIT
│   ├── styles/ (theme.py)                  ← ✅ FAIT
│   └── assets/ (logos)                     ← ✅ FAIT
├── k8s/                                    ← ❌ À FAIRE (dossier complet)
│   ├── namespaces.yaml
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── webapp-deployment.yaml
│   └── webapp-service.yaml
├── load_tests/                             ← ✅ FAIT
│   └── locustfile.py                       ← ✅ FAIT (3 scénarios de charge)
├── monitoring/                             ← ❌ À FAIRE (bonus)
│   └── prometheus.yml
├── tests/                                  ← ✅ FAIT
│   ├── conftest.py                         ← ✅ FAIT
│   ├── test_api.py                         ← ✅ FAIT
│   ├── test_preprocessing.py               ← ✅ FAIT
│   ├── test_model.py                       ← ✅ FAIT
│   └── test_pipeline.py                    ← ✅ FAIT
├── conception/                             ← ✅ FAIT
│   ├── conception.md                       ← ✅ FAIT (ce document)
│   ├── architecture.drawio                 ← ✅ FAIT
│   └── architecture.png                    ← ✅ FAIT
├── artifacts/                              ← ✅ FAIT (gitignored)
├── docker-compose.yml                      ← ✅ FAIT (9 services)
├── .env / .env.example                     ← ✅ FAIT
├── init-db.sql                             ← ✅ FAIT
├── pyproject.toml                          ← ✅ FAIT
├── Makefile                                ← ✅ FAIT
├── CLAUDE.md                               ← ✅ FAIT
├── .gitignore                              ← ✅ FAIT
└── README.md                               ← ✅ FAIT
```

---

## 5. Métriques à collecter

### Dans MLflow (enregistrées à chaque entraînement) — déjà en place
- `average_precision_score` : l'AUC-PR (voir section 3.2), métrique principale de comparaison entre modèles
- `f1_score` : moyenne harmonique de Precision et Recall (bon résumé en un seul chiffre)
- `precision_score` : "quand le modèle dit fraude, a-t-il raison ?"
- `recall_score` : "parmi les vraies fraudes, combien le modèle en a-t-il trouvé ?"
- `roc_auc_score` : aire sous la courbe ROC (loggée pour référence, mais on ne l'utilise pas pour comparer car trompeuse avec 0.17% de fraudes)

### Dans Prometheus/Grafana (collectées en temps réel par l'API) — ❌ À FAIRE
- `fraudguard_requests_total` : compteur du nombre de requêtes reçues (par endpoint et code de retour)
- `fraudguard_request_duration_seconds` : histogramme des temps de réponse (pour voir si l'API ralentit)
- `fraudguard_predictions_total` : compteur des prédictions (séparé "fraude" vs "légitime", pour détecter si le modèle se met à tout classifier pareil)
- `fraudguard_prediction_proba` : distribution des probabilités de fraude retournées (si la moyenne dérive dans le temps, c'est un signal de drift)

---

## 6. Répartition des rôles — Les 5 personnes

*Règle d'or de l'équipe : Avant de commencer, nous définissons ensemble le format exact (JSON) de la donnée en entrée et en sortie. Chacun "mock" (simule) le travail des autres pour avancer en parallèle.*

### 👤 Rôle 1 : Data Scientist (Le Cerveau)
**Mission principale :** Créer l'algorithme de détection, tracer les expérimentations et valider les modèles.
* **[Objectif 2]** Construire un modèle de classification : développer un modèle de ML/DL adapté à la détection de fraude.
* **[Objectif 5]** Suivre les modèles et les expériences avec MLFlow : enregistrer les performances (métriques, paramètres) et suivre les versions.
* **[Objectif 3]** Stocker le modèle sur un model registry : sauvegarder le modèle entraîné pour qu'il soit prêt à être déployé.
* **[Objectif 15 - BONUS]** Ajouter de l'évaluation / human in the loop pour labelliser / corriger vos prédictions.

### 👤 Rôle 2 : Ingénieur Backend (Le Guichetier)
**Mission principale :** Développer et packager l'API qui servira le modèle en temps réel.
* **[Objectif 6]** Développer une API : construire une API permettant de recevoir une transaction et de recevoir une prédiction (ex: avec FastAPI). *(Mocking: renvoyer une fausse prédiction en attendant le modèle du Rôle 1).*
* **[Objectif 9 - Partie 1]** Dockeriser votre API : créer le Dockerfile pour encapsuler l'application.
* **[Objectif 16 - BONUS]** Gestion des accès à l'API de Serving et dashboard par utilisateur de l'API.

### 👤 Rôle 3 : Data Engineer (Le Logisticien)
**Mission principale :** Automatiser les flux de données et l'entraînement continu.
* **[Objectif 1]** Extraire et prétraiter les données : nettoyer les données Kaggle et les préparer pour l'entraînement.
* **[Objectif 4]** Créer un pipeline de réentraînement avec Apache Airflow : mettre en place les pipelines pour mettre à jour le modèle.
* **[Objectif 8]** Gérer l'Entraînement Continue (CT) : ajouter un ou plusieurs DAG Airflow avec des triggers pour réentraîner et déployer automatiquement un nouveau modèle avec des checks de performance.
* **[Objectif 11 - BONUS]** Gérer des données en live : concevoir le script qui simule l'arrivée continue de nouvelles transactions.

### 👤 Rôle 4 : DevOps / Cloud Engineer (L'Architecte)
**Mission principale :** Industrialiser l'infrastructure, le code et le déploiement.
* **[Objectif 10]** Versionner et documenter le projet sur GitHub : gérer la structure de fichiers propre et la documentation claire (README).
* **[Objectif 9 - Partie 2]** Déployer l'API sur un cluster Kubernetes en local (Docker Desktop ou MiniKube).
* **[Objectif 9 - Partie 3]** Utilisez la CI/CD de GitHub Actions pour automatiser le processus.
* **[Objectif 14 - BONUS]** Gérer correctement le versionning / montée de version et rollback des modèles en production.

### 👤 Rôle 5 : Full-Stack / QA & Ops (La Vigie)
**Mission principale :** Créer l'interface utilisateur, stress-tester le système et surveiller sa santé.
* **[Objectif 7]** Créer une WebApp pour interagir avec le modèle : développer une interface utilisateur simple (ex: Gradio, Streamlit) pour visualiser les résultats des prédictions.
* **[Objectif 12 - BONUS]** Ajouter du monitoring pour visualiser l'ensemble des métriques (API, perfs du modèle) via Prometheus + Grafana.
* **[Objectif 13 - BONUS]** Vous pouvez ajouter des tests de montée en charge (ex: avec Locust) pour prouver que le cluster K8s encaisse le choc.
* **[Objectif 17 - BONUS]** Alerting par mail lors d'un réentraînement ou d'une erreur.

---

### 🔍 Analyse critique de cette répartition

**✅ Ce qui est bien dans cette répartition :**

- Les 17 objectifs de l'énoncé sont tous couverts et assignés — rien n'est oublié.
- Les bonus sont répartis de façon équilibrée (1 bonus par personne).
- Les rôles sont clairs et les noms (Le Cerveau, Le Guichetier, etc.) aident à comprendre la mission.
- La règle du "mock" est excellente pour travailler en parallèle (ex: le Rôle 2 peut renvoyer une fausse prédiction en attendant le vrai modèle du Rôle 1).

**⚠️ Points d'attention — à adapter vu le travail déjà fait par Ahmed :**

Le repo d'Ahmed couvre déjà une bonne partie des objectifs 1, 2, 3, 4, 5, 6 et 9 (partie Dockerfile). Concrètement :

- **Obj 1** (extraction/prétraitement) — ✅ Fait (tâche `ingest_and_preprocess`).
  → **Rôle 3** : pas besoin de le recréer, mais doit le **vérifier** et créer le DAG CT (Obj 8) qui est le vrai travail restant.

- **Obj 2** (modèle ML) — ✅ Fait (LightGBM + IsolationForest).
  → **Rôle 1** : pas besoin de recréer les modèles, mais doit **vérifier** le code, valider les hyperparamètres, et surtout s'occuper du `.env` manquant + de l'enrichissement de l'API avec `/metrics` Prometheus.

- **Obj 3** (model registry) — ✅ Fait (MLflow Production/Staging).
  → **Rôle 1** : vérifier que la promotion (= passage en Production) fonctionne bien.

- **Obj 4** (pipeline Airflow) — ✅ Fait (1 DAG existe).
  → **Rôle 3** : le DAG de training existe. Le vrai travail est le DAG CT (Obj 8).

- **Obj 5** (MLflow tracking) — ✅ Fait (params, metrics, artefacts).
  → **Rôle 1** : vérifier, pas recréer.

- **Obj 6** (API FastAPI) — ✅ Fait (/predict, /predict_batch, /health).
  → **Rôle 2** : pas besoin de recréer l'API, mais doit l'**enrichir** (ajouter `/metrics` pour Prometheus) et vérifier les Dockerfiles.

- **Obj 9 partie 1** (Dockerfile) — ✅ Fait (api/Dockerfile, airflow/Dockerfile).
  → **Rôle 2** : vérifier, pas recréer.

**En résumé** : les Rôles 1, 2 et 3 ont moins de création "from scratch" que prévu initialement. Leur travail se recentre sur la **vérification** du code existant, la **correction** des problèmes identifiés (surtout le `.env` manquant), et les **enrichissements** (endpoint `/metrics`, DAG CT).

Les Rôles 4 et 5 ne sont pas impactés car Kubernetes, CI/CD, WebApp et Monitoring sont entièrement à créer.

**💡 Recommandation : redistribuer la charge avec ces ajustements**

- **Rôle 1** (Data Scientist) — Tâches originales : créer modèle + MLflow + Registry.
  → Tâches ajustées : **vérifier** modèle + MLflow + Registry. **Créer** le `.env` (urgent, bloque tout le monde, inclure AWS credentials). **Configurer** MLflow → AWS S3 (changer `--artifacts-destination`). **Enrichir** l'API avec `/metrics` Prometheus. **Enrichir** le README final.

- **Rôle 2** (Backend) — Tâches originales : créer API + Dockerfile.
  → Tâches ajustées : **vérifier** API + Dockerfile. **Écrire les tests pytest** (test_api.py, test_model.py, test_preprocessing.py) — c'est un objectif évalué et personne ne le couvre dans la répartition originale.

- **Rôle 3** (Data Engineer) — Tâches originales : créer ingestion + pipeline + CT.
  → Tâches ajustées : **vérifier** le DAG existant. **Créer** le DAG CT (Obj 8) — c'est le vrai gros morceau.

- **Rôle 4** (DevOps) — Tâches originales : K8s + CI/CD + doc.
  → Tâches ajustées : pas de changement, tout est à créer. Ajouter le **Makefile**.

- **Rôle 5** (Full-Stack/QA) — Tâches originales : WebApp + monitoring + tests charge.
  → Tâches ajustées : pas de changement, tout est à créer.

**⚠️ Lacune importante : les tests pytest ne sont assignés à personne dans la répartition originale.** L'énoncé les mentionne explicitement dans les livrables ("l'ensemble de vos tests : tests unitaires, test d'intégration, test end-to-end"). Le Rôle 2, qui a moins de travail de création que prévu, est le mieux placé pour s'en charger.

---

## 7. Matrice complète objectifs × personnes

**Rôle 1 — Data Scientist :**
🔍 Obj 2 (modèle), 🔍 Obj 3 (registry), 🔍 Obj 5 (MLflow), 🔧 Obj 6 (/metrics), 🔧 Obj 10 (README), 🎯 Fix .env, 🌟 Obj 15 (bonus)

**Rôle 2 — Backend :**
🔍 Obj 6 (vérifier API), 🔍 Obj 9p1 (Dockerfiles), 🔧 Obj 10 (tests pytest), 🌟 Obj 16 (bonus)

**Rôle 3 — Data Engineer :**
🔍 Obj 1 (vérifier ingestion), 🔍 Obj 4 (vérifier DAG), 🎯 Obj 8 (créer DAG CT), 🌟 Obj 11 (bonus), 🌟 Obj 17 (bonus)

**Rôle 4 — DevOps :**
🎯 Obj 9 (K8s + CI/CD), 🔧 Obj 10 (CI/CD + Makefile), 🌟 Obj 14 (bonus)

**Rôle 5 — Full-Stack / QA :**
🎯 Obj 7 (Streamlit), 🔧 Obj 10 (slides), 🌟 Obj 12 (bonus), 🌟 Obj 13 (bonus), 🌟 Obj 17 (bonus)

**Légende :** 🔍 vérifier l'existant — 🔧 enrichir/améliorer — 🎯 créer from scratch — 🌟 bonus

---

## 8. Timeline de la journée

**Durée totale : 6h30 (8h30 → 15h). Pas de pause déjeuner. Présentation à 15h.**

```
08h30-09h00  SETUP + COORDINATION (30 min)
             → Rôle 1 crée le fichier .env et le push immédiatement
             → Tous : git pull, docker-compose up --build
             → Vérifier que les 4 services existants tournent
             → Tour de table rapide : qui fait quoi, quels ports, quel format JSON
             → Chacun crée sa branche Git (feature/streamlit, feature/k8s, etc.)

09h00-12h00  DÉVELOPPEMENT PARALLÈLE (3h — le gros du travail)
             → Rôle 1 : .env + /metrics Prometheus + vérification ML + README
             → Rôle 2 : vérification API/Dockerfiles + tests pytest
             → Rôle 3 : DAG CT + vérification DAG existant
             → Rôle 4 : manifests K8s + workflows CI/CD + Makefile
             → Rôle 5 : WebApp Streamlit + Dockerfile + monitoring

12h00-13h30  INTÉGRATION + DÉPLOIEMENT (1h30)
             → Merge des branches dans main (Pull Requests)
             → Résolution des conflits (docker-compose.yml — Rôle 1 arbitre)
             → Test end-to-end : docker-compose up → DAG → API → Streamlit
             → Rôle 4 : docker build local → kubectl apply -f k8s/
             → Rôle 3 : trigger le DAG CT, montrer qu'il fonctionne
             → Corriger les bugs d'intégration

13h30-14h30  DOCUMENTATION + SCREENSHOTS (1h)
             → Rôle 5 : screenshots de toutes les interfaces (Airflow, MLflow, Swagger, Streamlit, K8s)
             → Rôle 5 : finalise les slides de présentation
             → Rôle 1 : met à jour le README final avec screenshots
             → Tout le monde relit et valide le repo

14h30-15h00  RÉPÉTITION (30 min)
             → Dry run de la présentation (6 min chrono, 2 essais)
             → Ajuster le rythme, répartir qui parle quand

15h00        PRÉSENTATION OFFICIELLE (6 min exposé + 4 min questions)
```

---

## 9. Instructions pour Claude Code

Ce document est la **spécification complète** du projet. État actuel et prochaines étapes :

1. **Ne pas toucher** aux fichiers existants qui fonctionnent, sauf pour les enrichir là où c'est indiqué.

2. **Déjà créés** (ne plus recréer) :
   - ✅ `.env` + `.env.example` — variables d'environnement complètes
   - ✅ `webapp/` — application Streamlit complète (multi-pages, composants, API client, thème)
   - ✅ `airflow/dags/fraud_retraining_ct.py` — DAG CT @daily
   - ✅ `tests/` — test_api.py, test_preprocessing.py, test_model.py, test_pipeline.py, conftest.py
   - ✅ `Makefile` — targets test, lint, format, up, down
   - ✅ LocalStack S3 + pgAdmin dans docker-compose.yml (8 services)

3. **Reste à créer** (par ordre de priorité) :
   - ❌ `k8s/` (namespaces.yaml, api-deployment.yaml avec `imagePullPolicy: Never`, api-service.yaml, webapp-deployment.yaml, webapp-service.yaml)
   - ❌ `.github/workflows/` (ci.yml, cd.yml — CD fait `docker build` local + `kubectl apply`, **pas de push sur DockerHub**)
   - ❌ `monitoring/prometheus.yml`

4. **Reste à enrichir** :
   - ❌ `api/main.py` → ajouter l'endpoint `GET /metrics` avec `prometheus_client`
   - ❌ `api/pyproject.toml` → ajouter `prometheus_client` dans les dépendances
   - ❌ `docker-compose.yml` → ajouter les services `prometheus` et `grafana`

5. **Spécificités infra** :
   - **LocalStack S3** pour les artefacts MLflow (émulateur AWS local, pas de compte cloud nécessaire)
   - **Images Docker locales** pour K8s — pas de DockerHub, `imagePullPolicy: Never` dans les deployments
   - **Docker Desktop K8s** — partage le daemon Docker avec le host

6. **Conventions** : Python 3.11, linter ruff (line-length=100), pas de code superflu ("less is more" — les profs vérifient qu'on comprend ce qu'on fait), chaque fichier doit être fonctionnel.