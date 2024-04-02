{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6870b084-a671-41d8-9aff-66ffd86065b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The system cannot find the path specified.\n"
     ]
    },
    {
     "ename": "OSError",
     "evalue": "[WinError 1314] A required privilege is not held by the client: '/kaggle/input' -> '..\\\\input'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mOSError\u001b[0m                                   Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[19], line 24\u001b[0m\n\u001b[0;32m     21\u001b[0m os\u001b[38;5;241m.\u001b[39mmakedirs(KAGGLE_WORKING_PATH, \u001b[38;5;241m0o777\u001b[39m, exist_ok\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[0;32m     23\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m---> 24\u001b[0m   os\u001b[38;5;241m.\u001b[39msymlink(KAGGLE_INPUT_PATH, os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m..\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124minput\u001b[39m\u001b[38;5;124m'\u001b[39m), target_is_directory\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[0;32m     25\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mFileExistsError\u001b[39;00m:\n\u001b[0;32m     26\u001b[0m   \u001b[38;5;28;01mpass\u001b[39;00m\n",
      "\u001b[1;31mOSError\u001b[0m: [WinError 1314] A required privilege is not held by the client: '/kaggle/input' -> '..\\\\input'"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "from tempfile import NamedTemporaryFile\n",
    "from urllib.request import urlopen\n",
    "from urllib.parse import unquote, urlparse\n",
    "from urllib.error import HTTPError\n",
    "from zipfile import ZipFile\n",
    "import tarfile\n",
    "import shutil\n",
    "\n",
    "CHUNK_SIZE = 40960\n",
    "DATA_SOURCE_MAPPING = 'sms-spam-collection-a-more-diverse-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2660681%2F4558602%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240402%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240402T045147Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D191d91d976f5515a6116160b70f91d571db4609def0b7d3de5bd35c3f0d1d6547292f9354d126a4dafe0d740ecffb2afcfb47037481297a0ed419a1e612f01c4b1a8a9d04e4465f905f43d56dece0fc90541f0d504c24443d9b6fed21a8a2f2222bc249b035d3485f39364ba06b87e1cbd3d1f0436a27d8c17313b883e547ec225e0f9ea7ce797cfba21c4b51ebd7d8520612678ee7ad50346466382796d738b593ff7832ac2cc74697b25e8ba6ca69974b87e50caaf59f528066f434f05a86fda1514d4054409dfc004335856d0e557918e4a8c8817d69db876543c6f2812677777d2fc2b55923c218c66c780185ec8a18e428f4724d27d0069f9ec5f5ef059'\n",
    "\n",
    "KAGGLE_INPUT_PATH='/kaggle/input'\n",
    "KAGGLE_WORKING_PATH='/kaggle/working'\n",
    "KAGGLE_SYMLINK='kaggle'\n",
    "\n",
    "!umount /kaggle/input/ 2> /dev/null\n",
    "shutil.rmtree('/kaggle/input', ignore_errors=True)\n",
    "os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)\n",
    "os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)\n",
    "\n",
    "try:\n",
    "  os.symlink(KAGGLE_INPUT_PATH, os.path.join(\"..\", 'input'), target_is_directory=True)\n",
    "except FileExistsError:\n",
    "  pass\n",
    "try:\n",
    "  os.symlink(KAGGLE_WORKING_PATH, os.path.join(\"..\", 'working'), target_is_directory=True)\n",
    "except FileExistsError:\n",
    "  pass\n",
    "\n",
    "for data_source_mapping in DATA_SOURCE_MAPPING.split(','):\n",
    "    directory, download_url_encoded = data_source_mapping.split(':')\n",
    "    download_url = unquote(download_url_encoded)\n",
    "    filename = urlparse(download_url).path\n",
    "    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)\n",
    "    try:\n",
    "        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:\n",
    "            total_length = fileres.headers['content-length']\n",
    "            print(f'Downloading {directory}, {total_length} bytes compressed')\n",
    "            dl = 0\n",
    "            data = fileres.read(CHUNK_SIZE)\n",
    "            while len(data) > 0:\n",
    "                dl += len(data)\n",
    "                tfile.write(data)\n",
    "                done = int(50 * dl / int(total_length))\n",
    "                sys.stdout.write(f\"\\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded\")\n",
    "                sys.stdout.flush()\n",
    "                data = fileres.read(CHUNK_SIZE)\n",
    "            if filename.endswith('.zip'):\n",
    "              with ZipFile(tfile) as zfile:\n",
    "                zfile.extractall(destination_path)\n",
    "            else:\n",
    "              with tarfile.open(tfile.name) as tarfile:\n",
    "                tarfile.extractall(destination_path)\n",
    "            print(f'\\nDownloaded and uncompressed: {directory}')\n",
    "    except HTTPError as e:\n",
    "        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')\n",
    "        continue\n",
    "    except OSError as e:\n",
    "        print(f'Failed to load {download_url} to path {destination_path}')\n",
    "        continue\n",
    "\n",
    "print('Data source import complete.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a856c73-a433-4fba-b8bb-acd97dd1aaee",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.classify.scikitlearn import SklearnClassifier\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from wordcloud import WordCloud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "855b7380-e3e5-46a8-86ff-e163d6d8a64b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbbea6af-216c-4e2a-9ae5-268618a67d60",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../input/sms-spam-collection-a-more-diverse-dataset/train.csv').rename(columns={'sms':'text'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48ecd66c-0633-4fc3-b39d-b83dc3f3470c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eda78061-c46d-40f1-b6db-b10432e2d66a",
   "metadata": {},
   "outputs": [],
   "source": [
    "red_palette = sns.color_palette(\"Reds_r\", 2)\n",
    "red_palette_c = sns.color_palette(\"Reds_r\", as_cmap=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60c4191d-58db-4b94-b1a3-4c56beabaab1",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 6), dpi=300)\n",
    "plt.subplot(1, 2, 1)\n",
    "sns.set_palette(red_palette)\n",
    "sns.countplot(x='label', data=df)\n",
    "plt.title('Class Distribution')\n",
    "plt.xlabel('Class')\n",
    "plt.ylabel('Count')\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "sns.set_palette(red_palette)\n",
    "df['text_length'] = df['text'].apply(lambda x: len(x.split()))\n",
    "sns.boxplot(x='label', y='text_length', data=df)\n",
    "plt.title('Text Length Distribution')\n",
    "plt.xlabel('Class')\n",
    "plt.ylabel('Text Length')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ef8f6d2-f099-4c1b-9192-705f9c79e96f",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 6), dpi=300)\n",
    "sns.set_palette(red_palette)\n",
    "sns.histplot(data=df, x='text_length', hue='label', kde=True, element='step')\n",
    "plt.title('Text Length Distribution with KDE')\n",
    "plt.xlabel('Text Length')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8383998-359a-403e-8ca3-64d21a0e5d8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "ham_text = \" \".join(df[df['label'] == 0]['text'])\n",
    "spam_text = \" \".join(df[df['label'] == 1]['text'])\n",
    "\n",
    "ham_wordcloud = WordCloud(width=800, height=800, background_color='white', colormap=red_palette_c).generate(ham_text)\n",
    "spam_wordcloud = WordCloud(width=800, height=800, background_color='white', colormap=red_palette_c).generate(spam_text)\n",
    "\n",
    "ham_image = ham_wordcloud.to_array()\n",
    "spam_image = spam_wordcloud.to_array()\n",
    "\n",
    "plt.figure(figsize=(12, 6), dpi=600)\n",
    "\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.imshow(ham_image, interpolation='bilinear')\n",
    "plt.title('Ham Messages Word Cloud')\n",
    "plt.axis('off')\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.imshow(spam_image, interpolation='bilinear')\n",
    "plt.title('Spam Messages Word Cloud')\n",
    "plt.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7dcb43d-f76a-4877-a912-16a1e83c5c10",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_text(text):\n",
    "    words = word_tokenize(text) #Tokenization\n",
    "    words = [word.lower() for word in words if word.isalnum()] #to Lowercase\n",
    "    words = [word for word in words if word not in stopwords.words(\"english\")] #Remove Stopwords\n",
    "    return \" \".join(words) #Concate tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "35c83179-7be6-4685-8492-44f9576816b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text'] = df['text'].apply(preprocess_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d8927c7-3850-499a-9bab-f912c99d5035",
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))\n",
    "X = tfidf_vectorizer.fit_transform(df['text']).toarray()\n",
    "y = df['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f32ca88-b73d-49aa-a1e7-0168a36fd2c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "827202f5-a8cb-4fce-80af-073da2449b2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "sklearn_classifier = MultinomialNB(alpha=.1) #alpha=0.1 is more accurate for our model\n",
    "sklearn_classifier.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0848732-bbba-4ca3-89ab-00608dc85b9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "class SklearnNLTKClassifier(nltk.classify.ClassifierI): #Constructor\n",
    "    def __init__(self, classifier):\n",
    "        self._classifier = classifier\n",
    "\n",
    "    def classify(self, features): #Predict for one feature\n",
    "        return self._classifier.predict([features])[0]\n",
    "\n",
    "    def classify_many(self, featuresets): #Predict for multiple features\n",
    "        return self._classifier.predict(featuresets)\n",
    "\n",
    "    def prob_classify(self, features): #Shows error for not implementating\n",
    "        raise NotImplementedError(\"Probability estimation not available.\")\n",
    "\n",
    "    def labels(self): #return labels\n",
    "        return self._classifier.classes_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "540f664b-3e19-468e-9d5b-1fdb9b2568f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk_classifier = SklearnNLTKClassifier(sklearn_classifier)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27225f4e-e606-4ee4-999b-08c8426644f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = nltk_classifier.classify_many(X_test)\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "report = classification_report(y_test, y_pred)\n",
    "acc = f\"Accuracy is : {accuracy:.2f}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc2b9b86-ef5f-4e69-a79b-7e4fb01a51ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 6), dpi=300)\n",
    "plt.text(0.5, 0.6, report, fontsize=12, color='darkred', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='darkred'))\n",
    "plt.text(0.5, 0.4, acc, fontsize=12, color='Green', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='green'))\n",
    "plt.title('Classification Report')\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9dafc00-c0aa-4aa1-afa3-5e9832ed7816",
   "metadata": {},
   "outputs": [],
   "source": [
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(4, 3), dpi=200)\n",
    "sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])\n",
    "plt.title('CM of test data prediction')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.ylabel('True Label')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58e6ea51-dd87-4e18-8b05-a666d3d2818c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
