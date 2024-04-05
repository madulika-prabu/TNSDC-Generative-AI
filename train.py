classifiers = {
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Neural Network": MLPClassifier()
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(f"\n{name} Classification Report:\n{report}\n")

