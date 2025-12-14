#!/bin/bash
set -e

# Wait for postgres
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL started"

# Initialize Airflow database (only if not already initialized)
if [ ! -f "/opt/airflow/airflow.db" ]; then
  echo "Initializing Airflow database..."
  airflow db init

  # Create admin user
  echo "Creating admin user..."
  airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "User already exists"
fi

# Upgrade database (in case of schema changes)
airflow db upgrade

echo "Starting Airflow..."
exec "$@"
