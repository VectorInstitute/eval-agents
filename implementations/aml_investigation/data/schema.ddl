DROP VIEW IF EXISTS "account_transactions";

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
    FOREIGN KEY ("from_bank", "from_account")
        REFERENCES "accounts" ("bank_id", "account_number"),
    FOREIGN KEY ("to_bank", "to_account")
        REFERENCES "accounts" ("bank_id", "account_number")
);

CREATE VIEW account_transactions AS
SELECT
  transaction_id,
  timestamp,
  from_account AS account,
  'OUT' AS direction,
  to_account AS counterparty,
  from_bank AS bank,
  to_bank AS counterparty_bank,
  amount_paid AS amount,
  payment_currency AS currency,
  payment_format
FROM transactions
UNION ALL
SELECT
  transaction_id,
  timestamp,
  to_account AS account,
  'IN' AS direction,
  from_account AS counterparty,
  to_bank AS bank,
  from_bank AS counterparty_bank,
  amount_received AS amount,
  receiving_currency AS currency,
  payment_format
FROM transactions;
