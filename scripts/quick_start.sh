#!/bin/bash
# Quick start script for CoTA data generation.
# Option 1 is a mock smoke test. Production dataset generation requires real
# datasource paths and an API key-backed generator.

echo "🚀 Pixelis CoTA Data Generation - Quick Start"
echo "============================================"

# Check if running from project root
if [ ! -f "scripts/1_generate_specialized_datasets.py" ]; then
    echo "❌ Error: Please run this script from the Pixelis project root"
    exit 1
fi

# Check Python
echo "📦 Checking Python environment..."
python --version

# Test imports
echo "🧪 Testing system..."
python scripts/test_cota_generation.py
if [ $? -ne 0 ]; then
    echo "❌ System test failed. Please fix issues before continuing."
    exit 1
fi

echo ""
echo "✅ System ready! Choose an option:"
echo ""
echo "1) Run a small smoke test (10 samples, mock mode)"
echo "2) Generate real data (requires API key)"
echo "3) View documentation"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "🎭 Running mock generation test..."
        python scripts/1_generate_specialized_datasets.py \
            --manifest configs/data_generation_manifest.yaml \
            --output-dir data_outputs/test \
            --num-samples 10 \
            --verbose
        ;;
    2)
        # Check for API key
        if [ -z "$OPENROUTER_API_KEY" ]; then
            echo "⚠️  Warning: OPENROUTER_API_KEY not set"
            echo "Please set it with: export OPENROUTER_API_KEY='your-key'"
            echo "Or the system will use mock mode."
            echo ""
            read -p "Continue anyway? (y/n): " confirm
            if [ "$confirm" != "y" ]; then
                exit 0
            fi
        fi
        
        echo "🔥 Starting real data generation..."
        echo "Output directory: data_outputs/specialized"
        echo ""
        python scripts/1_generate_specialized_datasets.py \
            --manifest configs/data_generation_manifest.yaml \
            --output-dir data_outputs/specialized \
            --verbose
        ;;
    3)
        echo "📚 Opening documentation..."
        if [ -f "docs/COTA_GENERATION_GUIDE.md" ]; then
            cat docs/COTA_GENERATION_GUIDE.md | less
        else
            echo "Documentation not found at docs/COTA_GENERATION_GUIDE.md"
        fi
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✨ Done! Check the output directory for results."
