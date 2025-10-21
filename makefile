# ===========================================
# FSDStocks - Backtesting Automation Makefile
# ==========================================
PYTHON := python
SRC_DIR := fsdstocks
STOCKS_DIR := stocks
OUTPUT_DIR := $(SRC_DIR)/output
STRATEGY ?= ma_rsi_volatility
LIMIT ?=
QUIET ?= false

# Default year range for testing
YEARS := 2025 2024 2023 2022 2021 2020 2019 2018 2017 2016 2015 2014

# ===========================================
# Run backtest for a single stock CSV
# ===========================================
run:
	@if [ -z "$(file)" ]; then \
		echo "❌ Please specify a file, e.g. make run file=AAPL.csv"; \
		exit 1; \
	fi
	@echo ">>> Running backtest for $(file) using strategy '$(STRATEGY)'"
	@$(PYTHON) -m $(SRC_DIR).main $(STOCKS_DIR)/$(file) --strategy "$(STRATEGY)" --quiet $(QUIET)

# Example:
# make run file=AAPL.csv
# make run file=MSFT.csv STRATEGY=bollinger_rsi

# ===========================================
# Run backtests for multiple CSVs
# ===========================================
run-all:
	@echo ">>> Running backtests for up to $(LIMIT) CSVs in $(STOCKS_DIR) using strategy '$(STRATEGY)'"
	@echo "QUIET=$(QUIET) | LIMIT=$(LIMIT)"
	@mkdir -p $(OUTPUT_DIR)
	@if [ -n "$(LIMIT)" ]; then \
		files=$$(ls $(STOCKS_DIR)/*.csv | head -n $(LIMIT) | tr '\n' ' '); \
	else \
		files=$$(ls $(STOCKS_DIR)/*.csv | tr '\n' ' '); \
	fi; \
	echo "📊 Loading $$files"; \
	if [ "$(QUIET)" = "true" ]; then \
		$(PYTHON) -m $(SRC_DIR).main $$files --strategy "$(STRATEGY)" --quiet; \
	else \
		$(PYTHON) -m $(SRC_DIR).main $$files --strategy "$(STRATEGY)"; \
	fi


# Examples:
# make run-all STRATEGY=bollinger_rsi
# make run-all STRATEGY=momentum_breakout LIMIT=10
# make run-all QUIET=true LIMIT=5

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
# ===========================================
# Show available commands
# ===========================================
help:
	@echo ""
	@echo "FSDStocks Backtesting Commands:"
	@echo "--------------------------------"
	@echo "make run file=AAPL.csv STRATEGY=bollinger_rsi"
	@echo "make run-all STRATEGY=momentum_breakout LIMIT=10"
	@echo "make clean                 - Delete all output files"
	@echo "make test-all              - Clean and run all tests"
	@echo "make help                  - Show this message"