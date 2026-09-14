# CNBC candidate review protocol

Review the sampled title and URL without changing the deterministic filter
columns. Leave uncertain cases noted rather than forcing confidence.

## Relevant

The article's main subject is plausibly related to armed conflict,
international sanctions, trade conflict, geopolitical energy disruption,
major political instability, major-power diplomatic conflict, or
monetary/geoeconomic policy capable of affecting the US dollar.

## Not relevant

The matched term is incidental, metaphorical, or unrelated to the proposed
geopolitical/economic mechanism. Typical examples are sports, entertainment,
ordinary company news, consumer products, domestic lifestyle content, and
incidental keyword occurrences.

## Label fields

- `human_relevant`: enter `yes`, `no`, or `uncertain`.
- `human_primary_category`: enter one taxonomy category when relevant.
- `human_notes`: briefly explain ambiguous cases or filter errors.

Do not pre-fill judgments from an automated model. Agreement between reviewers
should be assessed before these labels are used as evaluation data.

