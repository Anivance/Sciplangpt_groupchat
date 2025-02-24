from setuptools import setup, find_packages

setup(
    name="sciplangpt_groupchat",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "sqlalchemy[asyncio]",  # Added asyncio extra
        "asyncpg",              # Added async PostgreSQL driver
        "psycopg2-binary",     # Keep this for backwards compatibility
        "redis",
        "faiss-cpu",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "numpy",
    ],
)