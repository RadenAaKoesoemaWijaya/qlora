# MongoDB initialization script
db = db.getSiblingDB('admin');

// Create application user
db.createUser({
  user: 'qlora_user',
  pwd: 'qlora_password',
  roles: [
    { role: 'readWrite', db: 'qlora_db' },
    { role: 'dbAdmin', db: 'qlora_db' }
  ]
});

// Create monitoring user
db.createUser({
  user: 'monitoring_user',
  pwd: 'monitoring_password',
  roles: [
    { role: 'clusterMonitor', db: 'admin' },
    { role: 'read', db: 'local' }
  ]
});

// Switch to application database
use qlora_db;

// Create collections with validation
db.createCollection('users', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['username', 'email', 'hashed_password', 'role'],
      properties: {
        username: {
          bsonType: 'string',
          description: 'Username must be a string'
        },
        email: {
          bsonType: 'string',
          pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$',
          description: 'Email must be a valid email address'
        },
        role: {
          enum: ['admin', 'trainer', 'viewer'],
          description: 'Role must be admin, trainer, or viewer'
        },
        created_at: {
          bsonType: 'date',
          description: 'Creation timestamp'
        },
        updated_at: {
          bsonType: 'date',
          description: 'Update timestamp'
        }
      }
    }
  }
});

db.createCollection('training_jobs', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['job_id', 'user_id', 'model_id', 'status', 'config'],
      properties: {
        job_id: {
          bsonType: 'string',
          description: 'Job ID must be a string'
        },
        user_id: {
          bsonType: 'string',
          description: 'User ID must be a string'
        },
        model_id: {
          bsonType: 'string',
          description: 'Model ID must be a string'
        },
        status: {
          enum: ['pending', 'running', 'completed', 'failed', 'cancelled'],
          description: 'Status must be a valid job status'
        },
        config: {
          bsonType: 'object',
          description: 'Training configuration'
        },
        progress: {
          bsonType: 'object',
          description: 'Training progress information'
        },
        created_at: {
          bsonType: 'date',
          description: 'Creation timestamp'
        },
        updated_at: {
          bsonType: 'date',
          description: 'Update timestamp'
        }
      }
    }
  }
});

db.createCollection('api_keys', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['key', 'user_id', 'name', 'permissions'],
      properties: {
        key: {
          bsonType: 'string',
          description: 'API key must be a string'
        },
        user_id: {
          bsonType: 'string',
          description: 'User ID must be a string'
        },
        name: {
          bsonType: 'string',
          description: 'Key name must be a string'
        },
        permissions: {
          bsonType: 'array',
          description: 'API key permissions'
        },
        is_active: {
          bsonType: 'bool',
          description: 'Key active status'
        },
        expires_at: {
          bsonType: 'date',
          description: 'Expiration timestamp'
        },
        created_at: {
          bsonType: 'date',
          description: 'Creation timestamp'
        }
      }
    }
  }
});

// Create indexes
db.users.createIndex({ username: 1 }, { unique: true });
db.users.createIndex({ email: 1 }, { unique: true });
db.training_jobs.createIndex({ job_id: 1 }, { unique: true });
db.training_jobs.createIndex({ user_id: 1 });
db.training_jobs.createIndex({ status: 1 });
db.training_jobs.createIndex({ created_at: -1 });
db.api_keys.createIndex({ key: 1 }, { unique: true });
db.api_keys.createIndex({ user_id: 1 });

// Insert default admin user (password: admin123)
db.users.insertOne({
  username: 'admin',
  email: 'admin@qlora.local',
  hashed_password: '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PJPh.u',  // admin123
  role: 'admin',
  created_at: new Date(),
  updated_at: new Date()
});

print('MongoDB initialization completed successfully');