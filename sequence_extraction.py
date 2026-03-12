import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python sequence_extraction_final.py PDB1 PDB2 PDB3 ...", file=sys.stderr)
    sys.exit(1)

pdb_codes = sys.argv[1:]

for code in pdb_codes:
    try:
        url = f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{code}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()[code.lower()]

        if data and data[0] and 'sequence' in data[0]:
            sequence = data[0]['sequence']
            print(f'{code} {sequence}')
        else:
            print(f'{code}: Sequence data not found.')

    except requests.exceptions.HTTPError as e:
        print(f'{code}: Error - HTTP Request failed ({e})', file=sys.stderr)
    except KeyError:
        print(f'{code}: Error - Data not found for PDB code in URL response.', file=sys.stderr)
    except Exception as e:
        print(f'{code}: An unexpected error occurred: {e}', file=sys.stderr)

