# Imports
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
import io
from PIL import Image

# Global Variables
df = None
processed_df = None
X, y = None, None
problem_type = None

# Loading Data
def load_sample_data(dataset_name):
    global df, processed_df
    try:
        if dataset_name == "Iris Dataset":
            data = load_iris()
        elif dataset_name == "Diabetes Dataset":
            data = load_diabetes()
        elif dataset_name == "Wine Dataset":
            data = load_wine()
        elif dataset_name == "Breast Cancer Dataset":
            data = load_breast_cancer()
        else:
            return None, "Unknown dataset"

        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        processed_df = df.copy()
        return df.head(), f"Loaded {dataset_name}"
    except Exception as e:
        return None, f"Failed to load sample data: {e}"


def load_data(file):
    global df, processed_df
    try:
        if file is None:
            return None, "No file uploaded."
        df = pd.read_csv(file.name)
        processed_df = df.copy()
        return df.head(), "Dataset loaded successfully!"
    except Exception as e:
        return None, f"Error loading data: {e}"

def summarize_data():
    global df
    if df is None:
        return "Upload a dataset first."

    buffer = io.StringIO()
    buffer.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        buffer.write("Missing Values:\n")
        for col, count in missing[missing > 0].items():
            buffer.write(f" - {col}: {count} missing ({count / len(df):.1%})\n")
    else:
        buffer.write("No missing values found.\n")

    buffer.write("\nData Types:\n")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        buffer.write(f" - {dtype}: {count} columns\n")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        buffer.write("\nNumerical Column Stats:\n")
        stats = df[num_cols].describe().T[['mean', 'std', 'min', 'max']]
        for col, row in stats.iterrows():
            buffer.write(f" - {col}: mean={row['mean']:.2f}, std={row['std']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}\n")

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        for col in cat_cols:
            if df[col].nunique() < 15:
                buffer.write(f"\nClass Distribution for '{col}':\n")
                for val, count in df[col].value_counts().items():
                    buffer.write(f" - {val}: {count} ({count / len(df):.1%})\n")

    return buffer.getvalue()

# Data Summarizing

# Data Visualizing

def visualize_data(plot_type, x_axis, y_axis, hue):
    global df, processed_df
    data = processed_df if processed_df is not None else df

    if data is None:
        return None, "Upload a dataset first."

    x_axis = x_axis.strip() if x_axis else None
    y_axis = y_axis.strip() if y_axis else None
    hue = hue.strip() if hue else None
    if hue in [None, "", "None"]:
        hue = None

    cols = data.columns.tolist()
    if plot_type not in ["Correlation Heatmap", "Missing Values", "Pair Plot"]:
        if not x_axis or x_axis not in cols:
            return None, f"X-axis column '{x_axis}' not found in dataset."
        if plot_type == "Scatter Plot" and (not y_axis or y_axis not in cols):
            return None, f"Y-axis column '{y_axis}' not found in dataset."
        if hue and hue not in cols:
            return None, f"Hue column '{hue}' not found in dataset."

    try:
        plt.figure(figsize=(10, 6))

        if plot_type == "Correlation Heatmap":
            numerical_df = data.select_dtypes(include=["number"])
            if numerical_df.empty:
                return None, "No numerical columns for correlation heatmap."
            sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")

        elif plot_type == "Scatter Plot":
            if hue:
                sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=hue)
            else:
                sns.scatterplot(data=data, x=x_axis, y=y_axis)
            plt.title(f"Scatter Plot: {x_axis} vs {y_axis}")

        elif plot_type == "Box Plot":
            if hue:
                sns.boxplot(data=data, y=x_axis, x=hue)
            else:
                sns.boxplot(data=data, y=x_axis)
            plt.title(f"Box Plot: {x_axis}")

        elif plot_type == "Histogram":
            if hue:
                sns.histplot(data=data, x=x_axis, hue=hue, kde=True)
            else:
                sns.histplot(data=data, x=x_axis, kde=True)
            plt.title(f"Histogram: {x_axis}")

        elif plot_type == "Count Plot":
            if pd.api.types.is_numeric_dtype(data[x_axis]) and data[x_axis].nunique() > 30:
                return None, "Count plots work best with categorical or discrete numerical data with few unique values."
            if hue:
                sns.countplot(data=data, x=x_axis, hue=hue)
            else:
                sns.countplot(data=data, x=x_axis)
            plt.title(f"Count Plot: {x_axis}")
            plt.xticks(rotation=45)

        elif plot_type == "Pair Plot":
            numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
            if not numerical_columns:
                return None, "No numerical columns for pairplot."
            cols_to_plot = numerical_columns[:5]
            if hue and hue in data.columns and data[hue].nunique() <= 10:
                plot_cols = cols_to_plot + [hue] if hue not in cols_to_plot else cols_to_plot
                sns.pairplot(data[plot_cols], hue=hue)
            else:
                sns.pairplot(data[cols_to_plot])

        elif plot_type == "Missing Values":
            if data.isnull().sum().sum() == 0:
                return None, "No missing values found."
            sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
            plt.title("Missing Values")

        else:
            return None, "Invalid plot type."

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        img = Image.open(buf)
        return img, None


    except Exception as e:
        return None, f"Plot error: {e}"


# Column Preprocessing
def preprocess_column(column_name, method):
    global df, processed_df

    if df is None:
        return None, "Upload a dataset first."

    if processed_df is None:
        processed_df = df.copy()

    column_name = column_name.strip() if column_name else None
    if not column_name or column_name not in processed_df.columns:
        return processed_df.head(), f"Column '{column_name}' not found."

    try:
        col_data = processed_df[column_name]

        if method == "Fill NA with Mean":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), f"Mean fill only for numeric columns."
            processed_df[column_name] = col_data.fillna(col_data.mean())

        elif method == "Fill NA with Median":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), f"Median fill only for numeric columns."
            processed_df[column_name] = col_data.fillna(col_data.median())

        elif method == "Fill NA with Mode":
            mode_val = col_data.mode()
            if mode_val.empty:
                return processed_df.head(), "Cannot find mode."
            processed_df[column_name] = col_data.fillna(mode_val[0])

        elif method == "Drop NA Rows":
            before_len = len(processed_df)
            processed_df = processed_df.dropna(subset=[column_name])
            dropped = before_len - len(processed_df)
            return processed_df.head(), f"Dropped {dropped} rows with NA in '{column_name}'."

        elif method == "Remove Outliers with ZScore":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), "Z-score outlier removal only works on numeric columns."

            col_non_na = col_data.dropna()
            mean = col_non_na.mean()
            std = col_non_na.std()

            if std == 0:
                return processed_df.head(), f"No variation in '{column_name}' to compute Z-score."

            z_scores = (col_non_na - mean).abs() / std
            filtered_indices = z_scores[z_scores < 3].index
            before_len = len(processed_df)
            processed_df = processed_df.loc[filtered_indices]
            dropped = before_len - len(processed_df)
            return processed_df.head(), f"Removed {dropped} outlier rows from '{column_name}' using Z-Score."

        elif method == "Remove Outliers with IQR":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), "IQR outlier removal only works on numeric columns."

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            mask = (col_data >= Q1 - 1.5 * IQR) & (col_data <= Q3 + 1.5 * IQR)
            before_len = len(processed_df)
            processed_df = processed_df[mask]
            dropped = before_len - len(processed_df)
            return processed_df.head(), f"Removed {dropped} outlier rows from '{column_name}' using IQR."

        elif method == "Label Encoding":
            if pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() > 20:
                return processed_df.head(), f"Warning: label encoding on numeric with many unique vals."
            le = LabelEncoder()
            processed_df[column_name] = le.fit_transform(col_data.astype(str))

        elif method == "One-Hot Encoding":
            if col_data.nunique() > 20:
                return processed_df.head(), f"Warning: one-hot encoding on column with many categories."
            dummies = pd.get_dummies(col_data, prefix=column_name, drop_first=True)
            processed_df = pd.concat([processed_df.drop(columns=[column_name]), dummies], axis=1)

        elif method == "Standard Scaling":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), f"Scaling only for numeric columns."
            scaler = StandardScaler()
            processed_df[column_name] = scaler.fit_transform(col_data.values.reshape(-1, 1))

        elif method == "Min-Max Scaling":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), f"Scaling only for numeric columns."
            scaler = MinMaxScaler()
            processed_df[column_name] = scaler.fit_transform(col_data.values.reshape(-1, 1))

        elif method == "Log Transform":
            if not pd.api.types.is_numeric_dtype(col_data):
                return processed_df.head(), f"Log transform only for numeric columns."
            if (col_data <= 0).any():
                return processed_df.head(), "Cannot log transform zero or negative values."
            processed_df[column_name] = np.log(col_data + 1e-10)

        else:
            return processed_df.head(), "Unknown preprocessing method."

        return processed_df.head(), f"Applied '{method}' on '{column_name}'."

    except Exception as e:
        return processed_df.head() if processed_df is not None else None, f"Error: {e}"


def reset_preprocessing():
    global df, processed_df
    if df is None:
        return None, "No dataset to reset."
    processed_df = df.copy()
    return processed_df.head(), "Reset to original data."

# Train & Test Preparing
def prepare_train_test(target_column):
    global df, processed_df, X, y, problem_type

    data = processed_df if processed_df is not None else df
    if data is None:
        return "Upload and preprocess data first."

    target_column = target_column.strip() if target_column else None
    if not target_column or target_column not in data.columns:
        return f"Target column '{target_column}' not found."

    # Fill missing
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                mode = data[col].mode()
                data[col] = data[col].fillna(mode[0] if not mode.empty else "Unknown")

    # Encode categorical
    for col in data.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if pd.api.types.is_numeric_dtype(y) and y.nunique() >= 15:
        problem_type = "Regression"
    else:
        problem_type = "Classification"

    return f"Prepared data for {problem_type}. Features: {X.shape[1]}, Samples: {X.shape[0]}"


# Algorithms recommendation
def recommend_algorithms(target_column):
    global df, processed_df
    data = processed_df if processed_df is not None else df
    if data is None:
        return "Upload and preprocess data first.", []

    target_column = target_column.strip() if target_column else None
    if not target_column or target_column not in data.columns:
        return f"Target column '{target_column}' not found.", []

    unique_vals = data[target_column].nunique()
    if pd.api.types.is_numeric_dtype(data[target_column]) and unique_vals >= 15:
        problem = "Regression"
    else:
        problem = "Classification"

    recommendations = []
    result = f"Problem Type: {problem}\n\nRecommended Algorithms:\n"

    if problem == "Classification":
        recommendations = ["Random Forest", "Gradient Boosting", "Decision Tree", "Logistic Regression", "SVM", "K-Nearest Neighbors"]
        result += "- Random Forest Classifier\n- Gradient Boosting Classifier\n- Decision Tree\n"
        if unique_vals == 2:
            result += "- Logistic Regression\n- SVM\n"
        result += "- K-Nearest Neighbors\n"

    else:
        recommendations = ["Random Forest", "Gradient Boosting", "Linear Regression", "SVM", "K-Nearest Neighbors", "Decision Tree"]
        result += "- Random Forest Regressor\n- Gradient Boosting Regressor\n- Linear Regression\n- SVR\n- K-Nearest Neighbors Regressor\n- Decision Tree Regressor\n"

    return result, recommendations


# Train & Evaluate
def train_evaluate_model(algorithm, test_size, n_estimators, max_depth, cv_folds):
    global X, y, problem_type

    if X is None or y is None:
        return "Prepare data for training first.", None

    try:
        test_size = float(test_size)
        n_estimators = int(n_estimators)
        cv_folds = int(cv_folds)
        max_depth = int(max_depth) if max_depth and str(max_depth).strip().isdigit() else None
    except ValueError:
        return "Invalid hyperparameters. Check your inputs.", None

    # Only use n_estimators and max_depth for models that support them
    if problem_type == "Classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            "Linear Regression": LinearRegression(),
            "SVM": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        }

    model = models.get(algorithm)
    if model is None:
        return f"Unknown algorithm '{algorithm}' for {problem_type}.", None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_train, y_train)

    # Robust cross-validation folds
    try:
        if problem_type == "Classification":
            min_class_count = np.min(np.bincount(y.astype(int))) if hasattr(y, 'astype') and len(np.unique(y)) > 1 else len(y)
            cv = min(cv_folds, len(y), min_class_count)
        else:
            cv = min(cv_folds, len(y))
        if cv < 2:
            cv = 2
        cv_scores = cross_val_score(model, X, y, cv=cv)
    except Exception:
        cv_scores = np.array([0])

    y_pred = model.predict(X_test)

    result = f"Model: {algorithm}\n\n"

    plt.figure(figsize=(8, 6))
    if problem_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        result += f"Test Accuracy: {acc:.4f}\nCross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\nClassification Report:\n{report}\n"
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        result += f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nCross-Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n"
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(0, color='red')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')

    # Feature importance if applicable
    try:
        if algorithm in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            result += "\nTop 5 Important Features:\n"
            for i in range(min(5, len(indices))):
                result += f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}\n"
    except AttributeError:
        pass

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img = Image.open(buf)
    return result, img

# GUI
with gr.Blocks(title="ManualML") as demo:
    gr.Markdown("# Machine Learning GUI")
    gr.Markdown("Upload your data, preprocess it, train & evaluate ML models — manual text input for column selections.")

    status_bar = gr.Textbox(label="Status Messages", value="Welcome! Start by loading a dataset.", interactive=False)

    with gr.Accordion("1. Data Loading", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Your Dataset")
                file_input = gr.File(label="Upload CSV File")
                load_btn = gr.Button("Load Data", variant="primary")

                gr.Markdown("### Or Use Sample Dataset")
                sample_dataset = gr.Dropdown(
                    ["Iris Dataset", "Diabetes Dataset", "Wine Dataset", "Breast Cancer Dataset"],
                    label="Select Sample Dataset",
                    value="Iris Dataset"
                )
                sample_btn = gr.Button("Load Sample Dataset")

            with gr.Column(scale=2):
                data_output = gr.Dataframe(label="Data Preview", interactive=False)

    with gr.Accordion("2. Data Analysis", open=False):
        with gr.Row():
            with gr.Column():
                summary_btn = gr.Button("Generate Dataset Summary")
                summary_output = gr.Textbox(label="Dataset Summary", lines=10, interactive=False)

            with gr.Column():
                gr.Markdown("### Data Visualization")
                plot_type = gr.Dropdown(
                    ["Correlation Heatmap", "Scatter Plot", "Box Plot", "Histogram", "Count Plot", "Pair Plot", "Missing Values"],
                    label="Select Plot Type",
                    value="Correlation Heatmap"
                )
                x_axis = gr.Textbox(label="X-axis Column (type column name)")
                y_axis = gr.Textbox(label="Y-axis Column (type column name, optional)", value="")
                hue_column = gr.Textbox(label="Grouping Variable (optional)", value="None")

                vis_btn = gr.Button("Generate Visualization", variant="primary")
                plot_output = gr.Image(label="Visualization Output")
                vis_status = gr.Textbox(label="Visualization Status", interactive=False)

    with gr.Accordion("3. Preprocessing", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Column-specific Preprocessing")
                column_selector = gr.Textbox(label="Column to Preprocess (type column name)")
                preprocess_method = gr.Dropdown(
                    [
                        "Fill NA with Mean", "Fill NA with Median", "Fill NA with Mode", "Drop NA Rows",
                        "Remove Outliers with ZScore", "Remove Outliers with IQR",
                        "Label Encoding", "One-Hot Encoding",
                        "Standard Scaling", "Min-Max Scaling", "Log Transform"
                    ],
                    label="Select Preprocessing Method",
                    value="Fill NA with Mean"
                )
                process_btn = gr.Button("Apply Preprocessing", variant="primary")
                reset_btn = gr.Button("Reset to Original Data")
                preprocess_status = gr.Textbox(label="Preprocessing Status", interactive=False)

            with gr.Column():
                preprocess_result = gr.Dataframe(label="Processed Data Preview", interactive=False)

    with gr.Accordion("4. Model Training", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Prepare Data for Training")
                target_column = gr.Textbox(label="Target Column (type column name)")
                prepare_btn = gr.Button("Prepare Data & Recommend Algorithms", variant="primary")
                preparation_status = gr.Textbox(label="Data Preparation Status", interactive=False)
                algorithm_recommendations = gr.Textbox(label="Algorithm Recommendations", lines=8, interactive=False)

            with gr.Column():
                gr.Markdown("### Model Configuration")
                algorithm_selector = gr.Radio(
                    ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM",
                     "K-Nearest Neighbors", "Decision Tree"],
                    label="Select Algorithm",
                    value="Random Forest"
                )

                test_size = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05,
                                      label="Test Set Size")
                cv_folds = gr.Slider(minimum=2, maximum=10, value=5, step=1,
                                     label="Cross-Validation Folds")
                n_estimators = gr.Slider(minimum=10, maximum=500, value=100, step=10,
                                         label="Number of Estimators (for ensemble methods)")
                max_depth = gr.Textbox(label="Max Depth (empty for None)", value="")

                train_btn = gr.Button("Train and Evaluate Model", variant="primary")

        with gr.Row():
            evaluation_output = gr.Textbox(label="Model Evaluation Results", lines=15, interactive=False)
            eval_plot = gr.Image(label="Evaluation Plot")

    # Event bindings
    load_btn.click(
        load_data,
        inputs=file_input,
        outputs=[data_output, status_bar]
    )

    sample_btn.click(
        load_sample_data,
        inputs=sample_dataset,
        outputs=[data_output, status_bar]
    )

    summary_btn.click(
        summarize_data,
        inputs=[],
        outputs=summary_output
    )

    vis_btn.click(
        visualize_data,
        inputs=[plot_type, x_axis, y_axis, hue_column],
        outputs=[plot_output, vis_status]
    )

    process_btn.click(
        preprocess_column,
        inputs=[column_selector, preprocess_method],
        outputs=[preprocess_result, preprocess_status]
    )

    reset_btn.click(
        reset_preprocessing,
        inputs=[],
        outputs=[preprocess_result, preprocess_status]
    )

    prepare_btn.click(
        prepare_train_test,
        inputs=[target_column],
        outputs=preparation_status
    ).then(
        recommend_algorithms,
        inputs=[target_column],
        outputs=[algorithm_recommendations, algorithm_selector]
    )

    train_btn.click(
        train_evaluate_model,
        inputs=[algorithm_selector, test_size, n_estimators, max_depth, cv_folds],
        outputs=[evaluation_output, eval_plot]
    )

# Run
if __name__ == "__main__":
    demo.launch()
