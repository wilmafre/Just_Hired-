import json
import os

#Set year
year = '2020'

# Open the input and output files
input_file = f'{year}.json'
output_file = f'{year}_clean.json'
input_size = os.path.getsize(input_file)

keys_to_remove = [
    "id",
    "external_id",
    "webpage_url",
    "logo_url",
    "headline",
    "application_deadline",
    "number_of_vacancies",
    "employment_type",
    "salary_type",
    "salary_description",
    "duration",
    "working_hours_type",
    "scope_of_work",
    "access",
    "employer",
    "application_details",
    "experience_required",
    "access_to_own_car",
    "driving_license_required",
    "driving_license",
    "occupation",
    "occupation_group",
    "occupation_field",
    "workplace_address",
    "must_have",
    "nice_to_have",
    "application_contacts",
    "last_publication_date",
    "removed",
    "removed_date",
    "source_type",
    "timestamp"
]

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    # Load the entire JSON data into memory
    data = json.load(fin)

    # Remove the unwanted keys from each item in the JSON data
    for item in data:
        for key in keys_to_remove:
            item.pop(key, None)

    # Write the modified JSON data to the output file
    json.dump(data, fout)

output_size = os.path.getsize(output_file)

print(f"Input file size: {input_size} bytes")
print(f"Output file size: {output_size} bytes")