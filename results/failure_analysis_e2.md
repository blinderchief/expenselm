# Failure analysis — e2 (200 examples, 67 fully correct)

| bucket | count | share of examples |
|---|---|---|
| schema_break | 38 | 19.0% |
| category_confusion | 1 | 0.5% |
| currency_error | 8 | 4.0% |
| date_resolution | 11 | 5.5% |
| merchant_error | 3 | 1.5% |
| policy_misapplication | 94 | 47.0% |

## schema_break (no_json_found) — 38 examples

**proc-01792**
- input: `got charged 319 rs for courier to the client at Blue Datt client visit, receipts shared on mail`
- predicted: `<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool`
- gold: `{"expenses": [{"amount": 319.0, "currency": "INR", "category": "other", "merchant": "Blue Dart", "date": null, "description": "courier to the client", "reimbursable": true, "policy_flags": []}], "conf`

**proc-01193**
- input: `like courier to the client ka 2830 rs with the Zentech clients, bhai Rs 4454 ka hotel near the client office at FabHotel Prime for the sprint demo day`
- predicted: `<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool_call>

<tool`
- gold: `{"expenses": [{"amount": 2830.0, "currency": "INR", "category": "other", "merchant": null, "date": null, "description": "courier to the client", "reimbursable": true, "policy_flags": ["over_limit", "m`

_Why (fill in by hand after reading the bucket):_

## category_confusion — 1 examples

**proc-00405**
- input: `so basically ksraoke night ka ₹3067, also bhai 1466 rupees ka wireless mouse at Croma on 12/03/2026`
- predicted: `</tool_call>

<tool_call>

{"expenses":[{"amount":3067.0,"currency":"INR","category":"meals","merchant":null,"date":null,"description":"karaoke night","reimbursable":true,"policy_flags":["over_limit",`
- gold: `{"expenses": [{"amount": 3067.0, "currency": "INR", "category": "entertainment", "merchant": null, "date": null, "description": "karaoke night", "reimbursable": false, "policy_flags": ["personal_expen`

_Why (fill in by hand after reading the bucket):_

## currency_error — 8 examples

**proc-01299**
- input: `quick one: put down ₹1,455 on confwrence registration. and metro card recharge came to 196 and quick one: spent Rs. 1407 on team lunch at Saravana Bhavan.`
- predicted: `</tool_call>

<tool_call>

{"expenses":[{"amount":1455.0,"currency":"INR","category":"other","merchant":null,"date":null,"description":"conference registration","reimbursable":true,"policy_flags":["ov`
- gold: `{"expenses": [{"amount": 1455.0, "currency": "INR", "category": "other", "merchant": null, "date": null, "description": "conference registration", "reimbursable": true, "policy_flags": ["missing_recei`

**proc-01656**
- input: `From: Sneha Iyer <sneha@acme corp.in>
Subject: reimbursement

Hi,

fyi shelled out ₹2375 for the beers for the team and put down 214 for
lunch

Thanks,
Sneha Iy`
- predicted: `</tool_call>

</tool_call>

{"expenses":[{"amount":2375.0,"currency":"INR","category":"meals","merchant":null,"date":null,"description":"beers for the team (alcohol)","reimbursable":false,"policy_flag`
- gold: `{"expenses": [{"amount": 2375.0, "currency": "INR", "category": "meals", "merchant": null, "date": null, "description": "beers for the team (alcohol)", "reimbursable": false, "policy_flags": ["persona`

_Why (fill in by hand after reading the bucket):_

## date_resolution — 11 examples

**proc-01851**
- input: `courier to the client at Blue Dart came to ₹3651 last saturday and printer cartridge — Rs. 1231 client visit last week and wireless mouse came to INR 662 on 12/`
- predicted: `</tool_call>

<tool_call>

{"expenses":[{"amount":3651.0,"currency":"INR","category":"other","merchant":"Blue Dart","date":"2026-03-13","description":"courier to the client","reimbursable":true,"polic`
- gold: `{"expenses": [{"amount": 3651.0, "currency": "INR", "category": "other", "merchant": "Blue Dart", "date": "2026-03-14", "description": "courier to the client", "reimbursable": true, "policy_flags": ["`

**proc-01711**
- input: `reimbursement request: wine
for the party, ₹1,486, on 12/03/2026., fyi paid 590 for the courier to the client at Blue Dart for the team, have the bill today`
- predicted: `</tool_call>

</tool_call>

{"expenses":[{"amount":1486.0,"currency":"INR","category":"meals","merchant":null,"date":"2026-03-12","description":"wine for the party (alcohol)","reimbursable":false,"pol`
- gold: `{"expenses": [{"amount": 1486.0, "currency": "INR", "category": "meals", "merchant": null, "date": "2026-03-12", "description": "wine for the party (alcohol)", "reimbursable": false, "policy_flags": [`

_Why (fill in by hand after reading the bucket):_

## merchant_error — 3 examples

**proc-00134**
- input: `yaar flight DEL-BLR at MakeMyTrip me ₹3,621 lag gaye + yaar Noyion team plan me 799 rupees lag gaye + bhai ₹1873 ka conference registration client workshop, rec`
- predicted: `</tool_call>

</tool_call>

{"expenses":[{"amount":3621.0,"currency":"INR","category":"travel","merchant":"MakeMyTrip","date":null,"description":"flight DEL-BLR","reimbursable":true,"policy_flags":["o`
- gold: `{"expenses": [{"amount": 3621.0, "currency": "INR", "category": "travel", "merchant": "MakeMyTrip", "date": null, "description": "flight DEL-BLR", "reimbursable": true, "policy_flags": ["missing_recei`

**proc-01968**
- input: `fyi spent ₹1596 for the one night stay at 0YO
Townhouse for the team, receipt attached`
- predicted: `</tool_call>

<tool_call>

{"expenses":[{"amount":1596.0,"currency":"INR","category":"lodging","merchant":null,"date":null,"description":"one night stay at OYO Townhouse","reimbursable":true,"policy_f`
- gold: `{"expenses": [{"amount": 1596.0, "currency": "INR", "category": "lodging", "merchant": "OYO Townhouse", "date": null, "description": "one night stay", "reimbursable": true, "policy_flags": []}], "conf`

_Why (fill in by hand after reading the bucket):_

## policy_misapplication — 94 examples

**proc-01299**
- input: `quick one: put down ₹1,455 on confwrence registration. and metro card recharge came to 196 and quick one: spent Rs. 1407 on team lunch at Saravana Bhavan.`
- predicted: `</tool_call>

<tool_call>

{"expenses":[{"amount":1455.0,"currency":"INR","category":"other","merchant":null,"date":null,"description":"conference registration","reimbursable":true,"policy_flags":["ov`
- gold: `{"expenses": [{"amount": 1455.0, "currency": "INR", "category": "other", "merchant": null, "date": null, "description": "conference registration", "reimbursable": true, "policy_flags": ["missing_recei`

**proc-00730**
- input: `From: Vikram Rao <vikram@nimbussoft.in>
Subject: expenses

Hi,

breakfast — ₹2354 for the team

Best,
Vikram Rao
Sent from my iPhone`
- predicted: `</tool_call>

<tool_call>

{"expenses":[{"amount":2354.0,"currency":"INR","category":"meals","merchant":null,"date":null,"description":"breakfast","reimbursable":true,"policy_flags":["over_limit"]}],"`
- gold: `{"expenses": [{"amount": 2354.0, "currency": "INR", "category": "meals", "merchant": null, "date": null, "description": "breakfast", "reimbursable": true, "policy_flags": ["over_limit", "missing_recei`

_Why (fill in by hand after reading the bucket):_
