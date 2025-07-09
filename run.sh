
#!/bin/bash

echo "Jacob: Share of Voice vs Market Share Analysis Platform"
echo "======================================================"
echo ""

# Check if config file exists
if [ ! -f "config/automotive.yaml" ]; then
    echo "Creating automotive configuration..."
    python -m src.cli init-config --config-template automotive --output config/automotive.yaml
fi

echo "Starting automotive industry analysis..."
python -m src.cli analyze --industry automotive --config config/automotive.yaml