from pycaret.classification import *
import pandas as pd
data = pd.read_csv('Student_performance_data _.csv')

clf = setup(data, target='GradeClass', session_id=123,
            numeric_features=['Age', 'StudyTimeWeekly', 'Absences'],
            categorical_features=['Gender', 'Ethnicity', 'ParentalEducation',
                                  'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'],
            ignore_features=['StudentID', 'GPA'])

best_model = compare_models()

save_model(best_model, 'student_performance_model')

print("Model training and saving completed successfully.")

