.PHONY: quality_check

RUST_LOG := RUST_LOG=info
PLOTS_DIR := plots
BENCHMARK_RESULTS := benchmark_results.png

quality_check:
	$(RUST_LOG) cargo run --release --bin quality_check
	cd $(PLOTS_DIR) && python3 main.py
	mv $(PLOTS_DIR)/$(BENCHMARK_RESULTS) ./$(BENCHMARK_RESULTS)

clean:
	rm -f $(BENCHMARK_RESULTS)
	rm -f benchmark_results.csv