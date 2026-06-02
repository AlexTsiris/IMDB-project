# Movie Rating Prediction (Part 2)

חיזוי הדירוג הממוצע (averageRating) של סרט לפני יציאתו לאקרנים, על בסיס נתונים הידועים מראש בלבד.

## תיאור המודל

המודל הטוב יותר הוא **Random Forest Regressor** (עטוף ב-Pipeline עם preprocessing מלא).
- **Target:** averageRating (בטווח 1.0 עד 10.0)
- **ביצועים (10-fold CV):** RMSE ≈ 1.13, MAE ≈ 0.87, R² ≈ 0.23
- **מודל שני להשוואה:** Elastic Net (RMSE ≈ 1.16)

ה-Pipeline כולל imputation, scaling, ו-OneHotEncoding. כל העיבוד המקדים מתבצע בתוך ה-cross-validation כדי למנוע data leakage.

## מבנה הקבצים

- `notebook.ipynb` : הקוד המלא, כולל prepare_data, אימון, הערכה, error analysis, fairness analysis, feature importance
- `model.pkl` : המודל המאומן (Random Forest), שמור עם joblib
- `requirements.txt` : רשימת הספריות והגרסאות
- `report.pdf` : הדוח המלא

## הוראות הרצה

```bash
# 1. התקנת הספריות
pip install -r requirements.txt

# 2. הרצת ה-notebook מקצה לקצה
jupyter notebook notebook.ipynb
```

## שימוש במודל המאומן (test-time)

```python
import pandas as pd
import joblib

df_2025 = pd.read_csv("test_set.csv")
X = prepare_data(df_2025)          # הפונקציה מוגדרת ב-notebook
model = joblib.load("model.pkl")
y_pred = model.predict(X)
```

> **הערה חשובה:** המודל נשמר עם scikit-learn **1.2.2** ו-Python **3.11**.
> כדי שטעינת `model.pkl` תעבוד ללא שגיאות, יש להשתמש באותן גרסאות (ראו requirements.txt).

## מניעת Data Leakage

ההגנה מיושמת בשלושה רבדים:
1. מחיקה מפורשת של עמודות leakage (averageRating, numVotes, BoxOffice) בתחילת `prepare_data`
2. רשימת whitelist של פיצ'רים מותרים בסוף `prepare_data`
3. כל ה-preprocessing מתבצע בתוך Pipeline שרץ רק על fold האימון ב-cross-validation

## Reproducibility

נעשה שימוש ב-`random_state=42` בכל המקומות הרלוונטיים (פיצול, מודלים, permutation).
