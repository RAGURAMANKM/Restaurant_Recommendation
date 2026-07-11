import pandas as pd
from sqlalchemy import create_engine, text
import pickle
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import streamlit as st




engine = create_engine(
    "postgresql+psycopg2://postgres:Lava90909@localhost:5432/swiggy_db"
)


print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("swiggy.csv")

print("Original Shape :", df.shape)



remove_columns = [
    "lic_no",
    "link",
    "address",
    "menu"
]

df.drop(columns=remove_columns, inplace=True, errors="ignore")

print("Unwanted Columns Removed")



duplicate_rows = df.duplicated().sum()
print("Duplicate Rows :", duplicate_rows)

df.drop_duplicates(inplace=True)



print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

df.dropna(inplace=True)



df["rating"] = (
    df["rating"]
    .astype(str)
    .str.replace("--", "", regex=False)
    .str.replace("NEW", "", regex=False)
    .str.strip()
)

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)



df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")
)

df["rating_count"] = pd.to_numeric(
    df["rating_count"],
    errors="coerce"
)



df["cost"] = (
    df["cost"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")
)

df["cost"] = pd.to_numeric(
    df["cost"],
    errors="coerce"
)



df.dropna(inplace=True)



df["id"] = df["id"].astype(int)
df["rating"] = df["rating"].astype(float)
df["rating_count"] = df["rating_count"].astype(int)
df["cost"] = df["cost"].astype(int)



df.reset_index(drop=True, inplace=True)

print("\nFinal Shape :", df.shape)

df.to_csv("cleaned_data.csv", index=False)

print("cleaned_data.csv Saved Successfully")



create_table = """
CREATE TABLE IF NOT EXISTS swiggy_restaurants
(
    id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    city VARCHAR(100),
    rating FLOAT,
    rating_count INTEGER,
    cost INTEGER,
    cuisine TEXT
)
"""

with engine.begin() as conn:
    conn.execute(text(create_table))

print("Table Ready")



count_query = "SELECT COUNT(*) FROM swiggy_restaurants"

row_count = pd.read_sql(count_query, engine).iloc[0, 0]

if row_count == 0:

    print("Inserting Data Into PostgreSQL...")

    df.to_sql(
        "swiggy_restaurants",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )

    print("Data Inserted Successfully")

else:

    print("Table Already Contains Data")
    print("Skipping Data Insertion")



count = pd.read_sql(
    "SELECT COUNT(*) AS total_rows FROM swiggy_restaurants",
    engine
)

print("\nTotal Rows In Database :", count.iloc[0]["total_rows"])

print("=" * 60)
print("PART 1 COMPLETED SUCCESSFULLY")
print("=" * 60)




engine = create_engine(
    "postgresql+psycopg2://postgres:Lava90909@localhost:5432/swiggy_db"
)



print("=" * 60)
print("Loading Data From PostgreSQL...")
print("=" * 60)

query = """
SELECT
id,
name,
city,
rating,
rating_count,
cost,
cuisine
FROM swiggy_restaurants
ORDER BY id
"""

df = pd.read_sql(query, engine)

print("Dataset Loaded Successfully")
print("Rows :", len(df))
print("Columns :", len(df.columns))



categorical_features = [
    "city",
    "cuisine"
]

numerical_features = [
    "rating",
    "rating_count",
    "cost"
]



preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)



print("\nEncoding Features...")

X = preprocessor.fit_transform(
    df[
        categorical_features +
        numerical_features
    ]
)

print("Encoding Completed")

print("Feature Matrix Shape :", X.shape)



sparse.save_npz(
    "restaurant_features.npz",
    X
)

print("restaurant_features.npz Saved")



with open("preprocessor.pkl", "wb") as file:
    pickle.dump(preprocessor, file)

print("preprocessor.pkl Saved")



print("\nTraining Nearest Neighbors Model...")

model = NearestNeighbors(
    n_neighbors=11,
    metric="cosine",
    algorithm="brute"
)

model.fit(X)

print("Model Training Completed")



with open("recommendation_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("recommendation_model.pkl Saved")



df.to_csv(
    "cleaned_data.csv",
    index=False
)

print("cleaned_data.csv Saved")



print("\nVerification")

print("Rows :", len(df))
print("Feature Matrix :", X.shape)
print("Number of Restaurants :", len(df))

print("=" * 60)
print("PART 2 COMPLETED SUCCESSFULLY")
print("=" * 60)




print("=" * 60)
print("Loading Files...")
print("=" * 60)

cleaned_data = pd.read_csv("cleaned_data.csv")

feature_matrix = sparse.load_npz("restaurant_features.npz")

with open("recommendation_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Files Loaded Successfully")



def recommend_restaurants(restaurant_name, top_n=10):

    restaurant_name = restaurant_name.strip().lower()

    matched = cleaned_data[
        cleaned_data["name"].str.lower() == restaurant_name
    ]

    if matched.empty:
        print("\nRestaurant Not Found")
        return pd.DataFrame()

    selected_index = matched.index[0]

    distances, indices = model.kneighbors(
        feature_matrix[selected_index],
        n_neighbors=top_n + 1
    )

    recommended = cleaned_data.iloc[
        indices[0][1:]
    ][
        [
            "name",
            "city",
            "rating",
            "rating_count",
            "cost",
            "cuisine"
        ]
    ].copy()

    recommended["Similarity"] = (
        (1 - distances[0][1:]) * 100
    ).round(2)

    return recommended.reset_index(drop=True)



print("\nAvailable Restaurant")

print(cleaned_data.loc[0, "name"])

restaurant = cleaned_data.loc[0, "name"]

print("\nSearching Recommendations For:")

print(restaurant)

recommendation = recommend_restaurants(
    restaurant,
    top_n=10
)

print("\nRecommended Restaurants")

print(recommendation)

print("=" * 60)
print("PART 3 COMPLETED SUCCESSFULLY")
print("=" * 60)




st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide"
)


engine = create_engine(
    "postgresql+psycopg2://postgres:Lava90909@localhost:5432/swiggy_db"
)


@st.cache_data
def load_restaurants():
    query = """
    SELECT
        id,
        name,
        city,
        rating,
        rating_count,
        cost,
        cuisine
    FROM swiggy_restaurants
    ORDER BY id
    """
    return pd.read_sql(query, engine)

restaurants = load_restaurants()



@st.cache_resource
def load_model():
    with open("recommendation_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()



@st.cache_resource
def load_features():
    return sparse.load_npz("restaurant_features.npz")

feature_matrix = load_features()



st.title("🍽️ Swiggy Restaurant Recommendation System")

st.write("Find restaurants similar to your favourite restaurant.")



restaurant_name = st.selectbox(
    "Select Restaurant",
    sorted(restaurants["name"].unique())
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


if st.button("Recommend Restaurants"):

    selected = restaurants[
        restaurants["name"] == restaurant_name
    ]

    if selected.empty:

        st.error("Restaurant Not Found")

    else:

        selected_index = selected.index[0]

        distances, indices = model.kneighbors(
            feature_matrix[selected_index],
            n_neighbors=top_n + 1
        )

        recommendations = restaurants.iloc[
            indices[0][1:]
        ].copy()

        recommendations["Similarity (%)"] = (
            (1 - distances[0][1:]) * 100
        ).round(2)

        recommendations = recommendations[
            [
                "name",
                "city",
                "rating",
                "rating_count",
                "cost",
                "cuisine",
                "Similarity (%)"
            ]
        ]

        st.success(f"{len(recommendations)} Restaurants Found")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )



st.divider()

st.subheader("Restaurants by City")

city = st.selectbox(
    "Select City",
    sorted(restaurants["city"].unique())
)

city_df = restaurants[
    restaurants["city"] == city
]

st.write(f"Total Restaurants : {len(city_df)}")

st.dataframe(
    city_df[
        [
            "name",
            "rating",
            "rating_count",
            "cost",
            "cuisine"
        ]
    ],
    use_container_width=True,
    hide_index=True
)



st.divider()

st.subheader("Top Rated Restaurants")

top_restaurants = restaurants.sort_values(
    by=["rating", "rating_count"],
    ascending=False
).head(10)

st.dataframe(
    top_restaurants[
        [
            "name",
            "city",
            "rating",
            "rating_count",
            "cost",
            "cuisine"
        ]
    ],
    use_container_width=True,
    hide_index=True
)



st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Restaurants",
        len(restaurants)
    )

with col2:
    st.metric(
        "Cities",
        restaurants["city"].nunique()
    )

with col3:
    st.metric(
        "Average Rating",
        round(restaurants["rating"].mean(), 2)
    )

with col4:
    st.metric(
        "Average Cost",
        f"₹{int(restaurants['cost'].mean())}"
    )



st.divider()

st.caption("Swiggy Restaurant Recommendation System using Machine Learning")