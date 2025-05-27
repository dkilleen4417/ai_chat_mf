# MongoDB Backup Scripts

This directory contains scripts for backing up and restoring MongoDB databases.

## Backup Script (`backup_mongodb.py`)

Creates a backup of a MongoDB database using `mongodump`.

### Prerequisites

- Python 3.6+
- MongoDB Database Tools (includes `mongodump`)
  - On macOS: `brew install mongodb-database-tools`

### Usage

```bash
# Basic usage (uses defaults)
python3 backup_mongodb.py

# With custom parameters
python3 backup_mongodb.py --db your_database_name --uri mongodb://your-connection-string --out /path/to/backups
```

### Parameters

- `--db`: Database name (default: `chat_mf`)
- `--uri`: MongoDB connection URI (default: `mongodb://localhost:27017`)
- `--out`: Output directory for backups (default: `./backups`)

### Example

```bash
# Create a backup of the production database
python3 backup_mongodb.py --db production_db --uri mongodb://user:password@production-db:27017 --out /backups/mongodb
```

## Restoring a Backup

To restore a backup, use the `mongorestore` command:

```bash
mongorestore --uri=mongodb://localhost:27017 /path/to/backup/directory
```

## Logs

Logs are written to `backup_mongodb.log` in the script's directory.
