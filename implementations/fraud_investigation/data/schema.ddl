DROP TABLE IF EXISTS "accounts";
CREATE TABLE "accounts" (
    "bank_name" TEXT,
    "bank_id" INTEGER,
    "account_number" TEXT,
    "entity_id" TEXT,
    "entity_name" TEXT,
    PRIMARY KEY ("bank_id", "account_number")
);

DROP TABLE IF EXISTS "transactions";
CREATE TABLE "transactions" (
    "transaction_id" TEXT PRIMARY KEY,
    "timestamp" Text,
    "date" TEXT,
    "day_of_week" TEXT,
    "time_of_day" TEXT,
    "from_bank" INTEGER,
    "from_account" TEXT,
    "to_bank" INTEGER,
    "to_account" TEXT,
    "amount_received" REAL,
    "receiving_currency" TEXT,
    "amount_paid" REAL,
    "payment_currency" TEXT,
    "payment_format" TEXT,
    "risk_score" REAL,
    FOREIGN KEY ("from_bank", "from_account")
        REFERENCES "accounts" ("bank_id", "account_number"),
    FOREIGN KEY ("to_bank", "to_account")
        REFERENCES "accounts" ("bank_id", "account_number")
);

DROP VIEW IF EXISTS "v_unified_transactions";
CREATE VIEW "v_unified_transactions" AS
SELECT
    t.*,
    -- Create a single unique ID for the sender
    t.from_bank || '_' || t.from_account as from_uid,
    -- Create a single unique ID for the receiver
    t.to_bank || '_' || t.to_account as to_uid
FROM transactions t;
