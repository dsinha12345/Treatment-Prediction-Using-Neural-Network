from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

options = {'age': ['30-34 years',
  '40-44 years',
  '35-39 years',
  '25-29 years',
  '21-24 years',
  '55-64 years',
  '45-49 years',
  '65 years +',
  '18-20 years',
  '50-54 years',
  '15-17 years',
  '12-14 years'],
 'arrests': ['none', 'once', 'Unknown', 'two or more times'],
 'dsm_diagnosis': ['Unknown',
  'opioid dependence',
  'other substance dependence',
  'cannabis dependence',
  'alcohol abuse',
  'alcohol dependence',
  'other substance abuse',
  'substance-induced disorder',
  'alcohol-induced disorder',
  'cannabis abuse',
  'cocaine dependence',
  'depressive disorders',
  'schizophrenia/other psychotic disorders',
  'bipolar disorders',
  'anxiety disorders',
  'other mental health condition',
  'opioid abuse',
  'cocaine abuse',
  'alcohol intoxication',
  'attention deficit/disruptive behavior disorders'],
 'education': ['1-3 years of college/vocational school',
  'grade 12 (or ged)',
  'grades 9 to 11',
  'kindergarten to grade 8',
  '4 years of college/postgraduate study',
  'Unknown'],
 'employment': ['unemployed',
  'part-time',
  'not in labor force',
  'full-time',
  'Unknown'],
 'ethnic': ['not of hispanic or latino origin',
  'Unknown',
  'mexican',
  'cuban or other specific hispanic',
  'puerto rican',
  'hispanic or latino'],
 'frequency_of_use_primary': ['daily use',
  'some use',
  'no use since last month',
  'Unknown'],
 'frequency_of_use_secondary': ['daily use',
  'some use',
  'Unknown',
  'no use since last month'],
 'frequency_of_use_tertiary': ['some use',
  'daily use',
  'Unknown',
  'no use since last month'],
 'age_at_first_use_primary': ['18-20 years',
  '25-29 years',
  '30 years and older',
  '12-14 years',
  '15-17 years',
  '11 and under',
  '21-24 years',
  'Unknown'],
 'age_at_first_use_secondary': ['25-29 years',
  '30 years and older',
  '15-17 years',
  '12-14 years',
  'Unknown',
  '18-20 years',
  '11 and under',
  '21-24 years'],
 'age_at_first_use_tertiary': ['11 and under',
  '12-14 years',
  '15-17 years',
  'Unknown',
  '21-24 years',
  '18-20 years',
  '30 years and older',
  '25-29 years'],
 'gender': ['female', 'male', 'Unknown'],
 'health_insurance_at_admission': ['Unknown',
  'medicaid',
  'private insurance, blue cross/blue shield, hmo',
  'medicare, other (e.g. tricare, champus)',
  'none'],
 'living_arrangements': ['independent living',
  'homeless',
  'dependent living',
  'Unknown'],
 'marital_status': ['now married',
  'divorced, widowed',
  'never married',
  'Unknown',
  'separated'],
 'medication_assisted_opioid_therapy': ['yes', 'no', 'Unknown'],
 'previous_substance_use_treatment_episodes': ['one prior treatment episodes',
  'no prior treatment episodes',
  'Unknown'],
 'source_of_income': ['other',
  'wages/salary',
  'retirement/pension, disability',
  'public assistance',
  'none',
  'Unknown'],
 'primary_source_of_payment': ['Unknown',
  'medicaid',
  'private insurance',
  'other',
  'self-pay',
  'other government payments',
  'no charge',
  'medicare'],
 'referral_source': ['other health care provider',
  'individual',
  'other community referral',
  'Unknown',
  'court/criminal justice referral/dui/dwi',
  'alcohol/drug use care provider',
  'employer/eap',
  'school (educational)'],
 'cooccurring_mental_and_substance_use_disorders': ['Unknown', 'no', 'yes'],
 'race': ['Indigenous', 'Other', 'White', 'Asian', 'Black', 'Unknown'],
 'route_of_administration_primary': ['injections',
  'smoking',
  'oral',
  'Unknown',
  'inhalation',
  'other'],
 'route_of_administration_secondary': ['injections',
  'smoking',
  'oral',
  'Unknown',
  'inhalation',
  'other'],
 'route_of_administration_tertiary': ['other',
  'smoking',
  'oral',
  'Unknown',
  'inhalation',
  'injections'],
 'type_of_treatment_service_setting': ['rehab/residential, long term (more than 30 days)',
  'ambulatory, non-intensive outpatient',
  'ambulatory, intensive outpatient',
  'rehab/residential, short term (30 days or fewer)',
  'rehab/residential, hospital (non-detox)',
  'detox, 24-hour, free-standing residential',
  'ambulatory, detoxification',
  'detox, 24-hour, hospital inpatient'],
 'state': ['MN',
  'MI',
  'MO',
  'OH',
  'ND',
  'IL',
  'IN',
  'IA',
  'SD',
  'KS',
  'WI',
  'NE'],
 'substance_use_primary': ['methamphetamine/speed',
  'heroin',
  'marijuana/hashish',
  'hallucinogens',
  'cocaine/crack',
  'Unknown',
  'other drugs',
  'other opiates and synthetics',
  'benzodiazepines',
  'pcp',
  'other stimulants',
  'non-prescription methadone',
  'other amphetamines',
  'other sedatives or hypnotics',
  'over-the-counter medications',
  'inhalants',
  'other tranquilizers',
  'barbiturates',
  'none'],
 'substance_use_secondary': ['heroin',
  'methamphetamine/speed',
  'alcohol',
  'none',
  'other opiates and synthetics',
  'other drugs',
  'marijuana/hashish',
  'cocaine/crack',
  'other sedatives or hypnotics',
  'Unknown',
  'benzodiazepines',
  'pcp',
  'over-the-counter medications',
  'hallucinogens',
  'other amphetamines',
  'inhalants',
  'other stimulants',
  'other tranquilizers',
  'non-prescription methadone',
  'barbiturates'],
 'substance_use_tertiary': ['marijuana/hashish',
  'benzodiazepines',
  'alcohol',
  'none',
  'other opiates and synthetics',
  'methamphetamine/speed',
  'heroin',
  'Unknown',
  'other drugs',
  'cocaine/crack',
  'other stimulants',
  'inhalants',
  'other amphetamines',
  'hallucinogens',
  'over-the-counter medications',
  'other sedatives or hypnotics',
  'non-prescription methadone',
  'other tranquilizers',
  'pcp',
  'barbiturates'],
 'veteran_status': ['Unknown', 'no', 'yes']}

categorical_features = list(options.keys())

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(len(categorical_features), 64)
        self.batch_norm1 = nn.BatchNorm1d(64)  # Batch normalization layer
        self.layer2 = nn.Linear(64, 32)
        self.batch_norm2 = nn.BatchNorm1d(32)  # Batch normalization layer
        self.layer3 = nn.Linear(32, 1)  # 1 output for binary classification

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.layer1(x)))
        x = torch.relu(self.batch_norm2(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x

model = LinearModel()
model.load_state_dict(torch.load('model.pth'))

with open('encoder_dict.pkl', 'rb') as f:
    trialdict = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {}
        for field, _ in options.items():
            user_input[field] = request.form[field]
        print(user_input)
        # Encode categorical features
        encoded_input = []
        for field, value in user_input.items():
            if field.lower() in categorical_features:
                encoded_value = trialdict[field].transform([value])[0]
                encoded_input.append(encoded_value)
            else:
                encoded_input.append(value)
        print(encoded_input)
        # Convert the input into a PyTorch tensor
        input_tensor = torch.tensor(encoded_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Make a prediction using the model
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.item()
        print(prediction)
        # Format the prediction result
        result = 'Treatment completed' if prediction > 0.5 else 'Treatment not completed'
        return render_template('result.html', prediction=result)

    return render_template('index.html', options=options)

if __name__ == '__main__':
    app.run(debug=True)
