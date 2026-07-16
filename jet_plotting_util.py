import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_feature_distributions(df, title, output_dir='processed_data'):
    """
    Plot feature distributions for QCD and TT jets and save the plots.

    Args:
        df (DataFrame): DataFrame containing features and labels
        title (str): Title for the plot
        output_dir (str): Directory to save the plots
    """
    # Fixed categorical color assignment (colorblind-safe, validated pair):
    # QCD is always blue, ttbar is always red, on every plot.
    class_colors = {0: '#2a78d6', 1: '#e34948'}
    class_labels = {0: 'QCD', 1: 'ttbar'}

    features = [col for col in df.columns if col != 'label']
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5*n_rows))

    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=df, x=feature, hue='label', bins=50, alpha=0.5,
                     palette=class_colors, hue_order=[0, 1])
        plt.title(feature)
        legend = plt.gca().get_legend()
        if legend is not None:
            legend.set_title(None)
            for text, label_key in zip(legend.get_texts(), [0, 1]):
                text.set_text(class_labels[label_key])

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved feature distribution plot to {plot_path}")

def plot_jet_image(image, title="Jet Image", output_dir='processed_data'):
    """
    Plot a single jet image
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='viridis')
    plt.colorbar(label='Energy (GeV)')
    plt.title(title)
    plt.xlabel('η')
    plt.ylabel('φ')

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved jet image plot to {plot_path}")

def plot_training_history(history, metrics=['loss', 'accuracy'], output_dir='processed_data', title='DNN Training History'):
    """
    Plot training history for neural networks
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    if len(metrics) == 1:
        axes = [axes]

    try:
        # this is the syntax for keras
        hist = history.history
    except:
        hist = history
    for ax, metric in zip(axes, metrics):
        ax.plot(hist[metric], label=f'Training {metric}', color='#2a78d6')
        ax.plot(hist[f'val_{metric}'], label=f'Validation {metric}', color='#1baf7a')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training history plot to {plot_path}")

def plot_loss_curve(evals_result, metric='logloss', output_dir='processed_data', title='Training Loss'):
    """
    Plot per-boosting-round train vs. validation loss from an XGBoost
    model's evals_result() (requires fitting with eval_set=[(X_train,
    y_train), (X_val, y_val)] so both series are present). The gap between
    the two curves -- val loss flattening or rising while train loss keeps
    falling -- is the overfitting signal.
    """
    # evals_result() keys are 'validation_0', 'validation_1', ... in the
    # order eval_set was passed, not named by train/val.
    train_key, val_key = list(evals_result.keys())[:2]
    train_loss = evals_result[train_key][metric]
    val_loss = evals_result[val_key][metric]
    rounds = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, train_loss, label='Train', color='#2a78d6', linewidth=2)
    plt.plot(rounds, val_loss, label='Validation', color='#1baf7a', linewidth=2)
    plt.xlabel('Boosting round')
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss curve plot to {plot_path}")

def plot_confusion_matrix(y_true, y_pred, labels=['QCD', 'TT'], output_dir='processed_data', title='Confusion Matrix'):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")

# plot roc curve
def plot_roc_curve(y_true, y_pred_proba, output_dir='processed_data', title='ROC Curve'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.6f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved ROC curve plot to {plot_path}")