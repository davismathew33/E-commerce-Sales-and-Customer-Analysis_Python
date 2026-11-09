"""
E-commerce Data Analysis (sales trends + RFM customer segmentation)
Input : online_retail_sample.csv  (or the real UCI Online Retail dataset)
Output: clean_sales.csv, rfm_segments.csv, dashboard_charts.png
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
FILE = "online_retail_sample.csv"   # for the real dataset: pd.read_excel("Online Retail.xlsx")
raw = pd.read_csv(FILE, parse_dates=["InvoiceDate"])

print("Raw rows:", len(raw))
print("Missing CustomerID:", raw["CustomerID"].isna().sum())
print("Duplicates:", raw.duplicated().sum())
print("Cancellations:", raw["InvoiceNo"].astype(str).str.startswith("C").sum())

# ---------------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------------
df = raw.dropna(subset=["CustomerID"]).drop_duplicates()
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()   # removes cancellations
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
print("Clean rows:", len(df))

# ---------------------------------------------------------------
# 3. KPIs
# ---------------------------------------------------------------
total_revenue = df["Revenue"].sum()
orders = df["InvoiceNo"].nunique()
customers = df["CustomerID"].nunique()
print(f"\nRevenue: {total_revenue:,.2f} | Orders: {orders} | "
      f"Customers: {customers} | Avg order value: {total_revenue / orders:,.2f}")

# ---------------------------------------------------------------
# 4. EXPLORATORY ANALYSIS
# ---------------------------------------------------------------
monthly = df.set_index("InvoiceDate").resample("MS")["Revenue"].sum()
top_products = df.groupby("Description")["Revenue"].sum().nlargest(10)
by_country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)

print("\nMonthly revenue:\n", monthly.round(0))
print("\nTop 10 products:\n", top_products.round(0))
print("\nRevenue by country:\n", by_country.round(0))

# ---------------------------------------------------------------
# 5. RFM SEGMENTATION
# ---------------------------------------------------------------
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
)

# Score each measure 1-4 (4 = best). Recency: fewer days = higher score.
rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)


def label_segment(row):
    if row["R"] >= 3 and row["F"] >= 3:
        return "Champions"
    if row["R"] >= 3:
        return "Recent / Promising"
    if row["F"] >= 3:
        return "At Risk"
    return "Lost / Low value"


rfm["Segment"] = rfm.apply(label_segment, axis=1)

segments = rfm.groupby("Segment").agg(
    Customers=("Monetary", "size"),
    Revenue=("Monetary", "sum"),
)
segments["Revenue %"] = (segments["Revenue"] / segments["Revenue"].sum() * 100).round(1)
print("\nRFM segments:\n", segments.round(0))

# ---------------------------------------------------------------
# 6. VISUALIZE
# ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

monthly.plot(ax=ax[0, 0], marker="o", title="Monthly Revenue")
top_products.sort_values().plot(kind="barh", ax=ax[0, 1], title="Top 10 Products by Revenue")
by_country.plot(kind="bar", ax=ax[1, 0], title="Revenue by Country")
segments["Customers"].plot(kind="pie", ax=ax[1, 1], autopct="%1.0f%%",
                           title="Customers by RFM Segment", ylabel="")

plt.tight_layout()
plt.savefig("dashboard_charts.png", dpi=110)
plt.show()

# ---------------------------------------------------------------
# 7. EXPORT RESULTS
# ---------------------------------------------------------------
df.to_csv("clean_sales.csv", index=False)
rfm.reset_index().to_csv("rfm_segments.csv", index=False)
print("\nSaved: clean_sales.csv, rfm_segments.csv, dashboard_charts.png")
