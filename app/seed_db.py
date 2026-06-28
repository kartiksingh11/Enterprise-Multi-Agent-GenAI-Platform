"""
app/seed_db.py

Creates data/enterprise.db (SQLite) with a small but realistic
multi-table schema + sample rows.

WHY HAND-WRITTEN DATA INSTEAD OF A LIBRARY LIKE Faker:
--------------------------------------------------------
- Zero extra dependencies (sqlite3 is in Python's standard library).
- Deterministic: every time you re-run this, you get the exact same
  rows, which makes debugging the SQL agent much easier — if the LLM's
  SQL output suddenly differs, you know it's the LLM, not the data,
  that changed.

SCHEMA
------
departments(id, name, budget)
employees(id, name, department_id, role, salary, hire_date)
customers(id, name, email, signup_date)
products(id, name, category, price)
orders(id, customer_id, product_id, quantity, order_date)

Run with:  python app/seed_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "enterprise.db")

SCHEMA_SQL = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL NOT NULL
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

DEPARTMENTS = [
    (1, "Engineering", 2_500_000),
    (2, "Sales", 1_200_000),
    (3, "Marketing", 800_000),
    (4, "HR", 400_000),
]

EMPLOYEES = [
    (1, "Aditi Sharma", 1, "Senior Engineer", 145000, "2021-03-15"),
    (2, "Rohan Mehta", 1, "Engineering Manager", 175000, "2019-07-01"),
    (3, "Priya Nair", 2, "Sales Lead", 110000, "2020-01-10"),
    (4, "Karan Verma", 2, "Account Executive", 85000, "2022-05-23"),
    (5, "Sneha Iyer", 3, "Marketing Manager", 98000, "2021-11-02"),
    (6, "Vikram Joshi", 4, "HR Generalist", 70000, "2022-09-14"),
    (7, "Anjali Rao", 1, "Junior Engineer", 95000, "2023-02-01"),
    (8, "Manish Gupta", 2, "Account Executive", 88000, "2023-06-19"),
]

CUSTOMERS = [
    (1, "Acme Corp", "contact@acme.com", "2022-01-05"),
    (2, "Globex Inc", "info@globex.com", "2022-03-12"),
    (3, "Initech", "hello@initech.com", "2023-02-20"),
    (4, "Umbrella LLC", "sales@umbrella.com", "2023-07-08"),
    (5, "Soylent Co", "team@soylent.com", "2024-01-15"),
]

PRODUCTS = [
    (1, "Cloud Storage Basic", "Software", 49.99),
    (2, "Cloud Storage Pro", "Software", 149.99),
    (3, "Analytics Dashboard", "Software", 299.99),
    (4, "Onboarding Service", "Services", 999.00),
    (5, "Priority Support Plan", "Services", 199.00),
]

ORDERS = [
    (1, 1, 1, 3, "2023-01-10"),
    (2, 1, 3, 1, "2023-02-15"),
    (3, 2, 2, 2, "2023-03-01"),
    (4, 3, 4, 1, "2023-04-22"),
    (5, 3, 1, 5, "2023-05-30"),
    (6, 4, 5, 1, "2023-06-18"),
    (7, 4, 2, 4, "2023-07-25"),
    (8, 5, 3, 2, "2024-01-20"),
    (9, 2, 1, 1, "2024-02-14"),
    (10, 5, 5, 1, "2024-03-09"),
]


def seed():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(SCHEMA_SQL)

    cur.executemany("INSERT INTO departments VALUES (?, ?, ?)", DEPARTMENTS)
    cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)", EMPLOYEES)
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", ORDERS)

    conn.commit()
    conn.close()
    print(f"Seeded database at {DB_PATH}")


if __name__ == "__main__":
    seed()
