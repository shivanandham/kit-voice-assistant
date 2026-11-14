CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
    id text PRIMARY KEY,
    owner_key text NOT NULL,
    role text NOT NULL,
    user_json jsonb,
    kit_token text,
    kit_token_expires_at timestamptz,
    created_at timestamptz DEFAULT now(),
    expires_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id uuid PRIMARY KEY,
    owner_key text NOT NULL,
    role text NOT NULL,
    title text,
    started_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id bigserial PRIMARY KEY,
    conversation_id uuid REFERENCES conversations ON DELETE CASCADE,
    role text NOT NULL,
    content text NOT NULL,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id bigserial PRIMARY KEY,
    conversation_id uuid,
    name text,
    input jsonb,
    output jsonb,
    is_error boolean,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pending_actions (
    id uuid PRIMARY KEY,
    session_id text,
    conversation_id uuid,
    tool_name text,
    args jsonb,
    summary text,
    status text DEFAULT 'pending',
    created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id int PRIMARY KEY,
    title text,
    type text,
    version text,
    effective_date date,
    applies_to jsonb,
    is_surprise boolean,
    text text,
    token_estimate int,
    content_hash text,
    active boolean DEFAULT true,
    fetched_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_chunks (
    id bigserial PRIMARY KEY,
    document_id int REFERENCES documents ON DELETE CASCADE,
    chunk_index int,
    heading text,
    content text,
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations (owner_key, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON doc_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation ON tool_calls (conversation_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at);

ALTER TABLE messages ADD COLUMN IF NOT EXISTS citations jsonb;

-- Citations (knowledge): every chunk carries a stable ref, its section or rule number and pages.
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS ref text;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS section text;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS page_start int;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS page_end int;

-- Daily abuse-limit counters, saved periodically so allowances survive a restart.
CREATE TABLE IF NOT EXISTS limit_counters (
    name text NOT NULL,
    key text NOT NULL,
    day date NOT NULL,
    count double precision NOT NULL,
    updated_at timestamptz DEFAULT now(),
    PRIMARY KEY (name, key, day)
);
