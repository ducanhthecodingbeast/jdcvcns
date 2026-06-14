#!/bin/bash
set -e

export TOP_K=9
export BGE_BATCH_SIZE=16
export VIRANKER_BATCH_SIZE=16
export MOCK_DATA_TEST=1
export VECTOR_BACKEND="${VECTOR_BACKEND:-pinecone}"
export PINECONE_HOST="${PINECONE_HOST:-http://localhost:15080}"

echo "Starting backend services (PostgreSQL, Pinecone Local)..."
for d in 1.0 2.0 3.0 4.0 6.0; do
    (cd "$d" && ../scripts/compose up -d postgres >/dev/null 2>&1 || true)
done
./scripts/pinecone_local >/dev/null 2>&1 || true
sleep 2

echo "Initializing database schemas..."
for port in 15410 15420 15430 15440 15600; do
    ./1.0/.venv/bin/python -c "import psycopg2; conn = psycopg2.connect(host='localhost', port=$port, user='jdcvcns', password='jdcvcns_dev_password', dbname='jdcvcns'); cur = conn.cursor(); cur.execute(open('1.0/demoAPI/schema.sql').read()); conn.commit()" >/dev/null 2>&1 || true
done

# Backup original Data and replace with test
echo "Swapping Data folder with test folder..."
if [ -d "Data_backup" ]; then
    echo "Data_backup already exists. Please resolve."
    exit 1
fi
if [ -d "Data" ]; then
    mv Data Data_backup
    DATA_EXISTED=true
else
    DATA_EXISTED=false
fi
mv test Data
cp Data/Data/cv.csv Data/mockcv.csv
cp Data/Data/cv.csv Data/cv.csv
cp Data/Data/jd.csv Data/jd.csv

# Trap to ensure we always restore Data even if a script fails
cleanup() {
    echo "Restoring original Data folder..."
    rm -f Data/mockcv.csv Data/cv.csv Data/jd.csv
    mv Data test
    if [ "$DATA_EXISTED" = true ]; then
        mv Data_backup Data
    fi
}
trap cleanup EXIT

echo "=========================================================="
echo "Running CV/JD Ranking Test Suite on 3 CVs and 9 JDs"
echo "=========================================================="

echo "[1/7] Running Phase 1.0 (AITeamVN Embedding)..."
./1.0/.venv/bin/python 1.0/dotpdtesting1.0.py > /dev/null

echo "[2/7] Running Phase 2.1 (AITeamVN Dense Only)..."
./1.0/.venv/bin/python 2.0/dotpdtesting2.1.py > /dev/null

echo "[3/7] Running Phase 2.2 (PhoBERT Dense Only)..."
./1.0/.venv/bin/python 2.0/dotpdtesting2.2.py > /dev/null

echo "[4/7] Running Phase 2.3 (Multilingual MiniLM)..."
./1.0/.venv/bin/python 2.0/dotpdtesting2.3.py > /dev/null

echo "[5/7] Running Phase 3.0 (BGE-M3 Pinecone Hybrid)..."
./1.0/.venv/bin/python 3.0/bgmewdranttesting3.0.py > /dev/null

echo "[6/7] Running Phase 4.0 (BGE-M3 + ViRanker)..."
./1.0/.venv/bin/python 4.0/virankertesting4.0.py > /dev/null

echo "[7/9] Running Phase 6.0 (JobBERT Cosine)..."
./1.0/.venv/bin/python 6.0/jobberttesting6.0.py > /dev/null

echo "[8/9] Running Phase 6.1 (JobBERT Dot Product)..."
./1.0/.venv/bin/python 6.0/jobberttesting6.1.py > /dev/null

echo "[9/9] Running Phase 6.2 (BM25)..."
./1.0/.venv/bin/python 6.0/bm25testing6.2.py > /dev/null

echo "=========================================================="
echo "TEST COMPLETED. FETCHING RANKING RESULTS FROM DATABASE:"
echo "=========================================================="
./1.0/.venv/bin/python test_rankings.py
./1.0/.venv/bin/python Data/export_drawio.py
