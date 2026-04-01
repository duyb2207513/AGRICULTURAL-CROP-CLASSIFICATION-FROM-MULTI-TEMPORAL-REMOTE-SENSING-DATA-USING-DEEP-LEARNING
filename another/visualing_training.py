import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_comparison():
    """
    Vẽ biểu đồ so sánh quá trình training giữa LSTM và Transformer
    """
    
    # Load histories
    lstm_history = None
    transformer_history = None
    
    if os.path.exists('training_history_lstm.json'):
        with open('training_history_lstm.json', 'r') as f:
            lstm_history = json.load(f)
        print("✓ Loaded LSTM training history")
    else:
        print("⚠️  LSTM history not found")
    
    if os.path.exists('training_history_transformer.json'):
        with open('training_history_transformer.json', 'r') as f:
            transformer_history = json.load(f)
        print("✓ Loaded Transformer training history")
    else:
        print("⚠️  Transformer history not found")
    
    if not lstm_history and not transformer_history:
        print("❌ No training history found!")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training Comparison: CNN+LSTM vs CNN+Transformer', 
                 fontsize=16, fontweight='bold')
    
    # Colors
    lstm_color = '#FF6B6B'
    transformer_color = '#4ECDC4'
    
    # ========== 1. TRAINING LOSS ==========
    ax = axes[0, 0]
    if lstm_history:
        epochs = range(1, len(lstm_history['train_loss']) + 1)
        ax.plot(epochs, lstm_history['train_loss'], 
               color=lstm_color, linewidth=2, label='LSTM', marker='o', markersize=3)
    
    if transformer_history:
        epochs = range(1, len(transformer_history['train_loss']) + 1)
        ax.plot(epochs, transformer_history['train_loss'], 
               color=transformer_color, linewidth=2, label='Transformer', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========== 2. TRAINING ACCURACY ==========
    ax = axes[0, 1]
    if lstm_history:
        epochs = range(1, len(lstm_history['train_acc']) + 1)
        ax.plot(epochs, lstm_history['train_acc'], 
               color=lstm_color, linewidth=2, label='LSTM', marker='o', markersize=3)
    
    if transformer_history:
        epochs = range(1, len(transformer_history['train_acc']) + 1)
        ax.plot(epochs, transformer_history['train_acc'], 
               color=transformer_color, linewidth=2, label='Transformer', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Accuracy (%)', fontsize=11)
    ax.set_title('Training Accuracy Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    # ========== 3. VALIDATION ACCURACY ==========
    ax = axes[1, 0]
    if lstm_history:
        epochs = range(1, len(lstm_history['val_acc']) + 1)
        ax.plot(epochs, lstm_history['val_acc'], 
               color=lstm_color, linewidth=2.5, label='LSTM', marker='o', markersize=4)
        # Mark best epoch
        best_idx = lstm_history['best_epoch'] - 1
        best_acc = lstm_history['best_acc']
        ax.plot(best_idx + 1, best_acc, 'r*', markersize=15, 
               label=f'Best LSTM: {best_acc:.2f}%')
    
    if transformer_history:
        epochs = range(1, len(transformer_history['val_acc']) + 1)
        ax.plot(epochs, transformer_history['val_acc'], 
               color=transformer_color, linewidth=2.5, label='Transformer', marker='s', markersize=4)
        # Mark best epoch
        best_idx = transformer_history['best_epoch'] - 1
        best_acc = transformer_history['best_acc']
        ax.plot(best_idx + 1, best_acc, 'g*', markersize=15, 
               label=f'Best Transformer: {best_acc:.2f}%')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title('Validation Accuracy Curve (MOST IMPORTANT)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    # ========== 4. LEARNING RATE ==========
    ax = axes[1, 1]
    if lstm_history:
        epochs = range(1, len(lstm_history['learning_rates']) + 1)
        ax.plot(epochs, lstm_history['learning_rates'], 
               color=lstm_color, linewidth=2, label='LSTM', marker='o', markersize=3)
    
    if transformer_history:
        epochs = range(1, len(transformer_history['learning_rates']) + 1)
        ax.plot(epochs, transformer_history['learning_rates'], 
               color=transformer_color, linewidth=2, label='Transformer', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: training_curves_comparison.png")
    plt.show()
    
    # ========== SUMMARY TABLE ==========
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    if lstm_history:
        print("\n🔴 LSTM Model:")
        print(f"  • Best Val Accuracy: {lstm_history['best_acc']:.2f}% (Epoch {lstm_history['best_epoch']})")
        print(f"  • Total Parameters: {lstm_history.get('total_params', 'N/A'):,}")
        print(f"  • Final Train Accuracy: {lstm_history['train_acc'][-1]:.2f}%")
        print(f"  • Final Loss: {lstm_history['train_loss'][-1]:.4f}")
    
    if transformer_history:
        print("\n🔵 Transformer Model:")
        print(f"  • Best Val Accuracy: {transformer_history['best_acc']:.2f}% (Epoch {transformer_history['best_epoch']})")
        print(f"  • Total Parameters: {transformer_history.get('total_params', 'N/A'):,}")
        print(f"  • Final Train Accuracy: {transformer_history['train_acc'][-1]:.2f}%")
        print(f"  • Final Loss: {transformer_history['train_loss'][-1]:.4f}")
    
    # Comparison
    if lstm_history and transformer_history:
        diff = transformer_history['best_acc'] - lstm_history['best_acc']
        print("\n" + "="*70)
        print("🎯 COMPARISON:")
        print("="*70)
        if diff > 0:
            print(f"✓ Transformer tốt hơn LSTM: +{diff:.2f}%")
        elif diff < 0:
            print(f"✓ LSTM tốt hơn Transformer: +{abs(diff):.2f}%")
        else:
            print(f"✓ Cả hai model có accuracy tương đương")
        
        print(f"\nParams difference: {transformer_history.get('total_params', 0) - lstm_history.get('total_params', 0):,} parameters")
    
    print("="*70)


def plot_individual_model(model_name='lstm'):
    """
    Vẽ chi tiết cho một model
    """
    filename = f'training_history_{model_name}.json'
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return
    
    with open(filename, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name.upper()} Training Analysis', fontsize=16, fontweight='bold')
    
    color = '#FF6B6B' if model_name == 'lstm' else '#4ECDC4'
    
    # Loss
    axes[0, 0].plot(history['train_loss'], color=color, linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)
    
    # Train Accuracy
    axes[0, 1].plot(history['train_acc'], color=color, linewidth=2)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].grid(alpha=0.3)
    
    # Val Accuracy
    axes[1, 0].plot(history['val_acc'], color=color, linewidth=2)
    best_idx = history['best_epoch'] - 1
    axes[1, 0].plot(best_idx, history['best_acc'], 'r*', markersize=15)
    axes[1, 0].set_title(f'Validation Accuracy (Best: {history["best_acc"]:.2f}%)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(history['learning_rates'], color=color, linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {model_name}_training_analysis.png")
    plt.show()


if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "TRAINING VISUALIZATION" + " "*26 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Plot comparison
    plot_training_comparison()
    
    # Ask if user wants individual plots
    print("\n" + "="*70)
    print("📈 Do you want individual detailed plots?")
    print("="*70)
    print("1. Yes, plot LSTM details")
    print("2. Yes, plot Transformer details")
    print("3. Yes, plot both")
    print("4. No, skip")
    
    # For automation, just plot both
    # In interactive mode, you can add input()
    
    print("\nPlotting both models for your thesis...")
    if os.path.exists('training_history_lstm.json'):
        plot_individual_model('lstm')
    
    if os.path.exists('training_history_transformer.json'):
        plot_individual_model('transformer')
    
    print("\n✅ Visualization complete!")
    print("📁 Generated files:")
    print("  • training_curves_comparison.png")
    if os.path.exists('training_history_lstm.json'):
        print("  • lstm_training_analysis.png")
    if os.path.exists('training_history_transformer.json'):
        print("  • transformer_training_analysis.png")