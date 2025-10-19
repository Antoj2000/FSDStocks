# ===========================================
# FSDStocks - Backtesting Automation Makefile
# ==========================================
PYTHON := python
SRC_DIR := fsdstocks
STOCKS_DIR := stocks
OUTPUT_DIR := $(SRC_DIR)/output

# Default year range for testing
YEARS := 2025 2024 2023 2022 2021 2020 2019 2018 2017 2016 2015 2014

# Run backtest for a single stock CSV
run:
	@echo ">>> Running backtest for $(file)"
	@$(PYTHON) -m $(SRC_DIR).main $(STOCKS_DIR)/$(file)

# Example: make run file=AAPL.csv

run-all:
	@echo ">>> Running backtests for all CSVs in $(STOCKS_DIR)..."
ifdef LIMIT
	$(PYTHON) -m $(SRC_DIR).main $(STOCKS_DIR)/*.csv --limit $(LIMIT)
else
	$(PYTHON) -m $(SRC_DIR).main $(STOCKS_DIR)/*.csv
endif
# Example : make run-all LIMIT=25
# Example: make run-all

# Re-run a full clean test cycle
test-all: clean run-all
	@echo ">>> Completed full clean test cycle."

# Example: make test-all

# Optional: clean command to remove outputs
clean:
	@echo ">>> Cleaning output directory..."
	@rm -rf $(OUTPUT_DIR)
	@echo "Done."

# ==============================
# Default help display
# ==============================
help:
	@echo ""
	@echo "FSDStocks Backtesting Commands:"
	@echo "--------------------------------"
	@echo "make run file=AAPL.csv     - Run backtest for a single ticker"
	@echo "make run-all               - Run backtest for all CSVs in stocks/"
	@echo "make clean                 - Delete all output files"
	@echo "make install               - Install Python dependencies"
	@echo "make test-all              - Clean and run all tests"
	@echo "make show                  - List generated outputs"
	@echo ""