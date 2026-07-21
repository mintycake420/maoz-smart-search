"""Small local Flask shell for the screen-share demo."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from .domain import ProfileValidationError
from .search import SearchEngine

# Form fields accepted by the demo add-profile endpoint, mapped onto the same
# Salesforce-shaped record the fixtures use.  The id and the synthetic markers are
# assigned server-side: a visitor can add a person, not forge a record.
_TEXT_LIMIT = 300
_NAME_LIMIT = 60


def create_app(engine: SearchEngine | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(MAX_CONTENT_LENGTH=32 * 1024)
    app.json.ensure_ascii = False
    search_engine = engine or SearchEngine.from_default()

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "synthetic_only": True, "network_inference": False})

    @app.get("/api/meta")
    def meta():
        # Facets come from the live snapshot, so a profile added a moment ago
        # appears in the filter lists immediately.  The golden set is deliberately
        # absent from this payload — and since the model-comparison exhibit was
        # removed, nothing in the runtime reads it at all: ranking, gating,
        # tiering, evidence and every response payload are golden-set-free.
        profiles, added = search_engine.directory_snapshot()
        return jsonify({
            "filters": {
                "sectors": sorted({profile.sector for profile in profiles if profile.sector}),
                "regions": sorted({profile.region for profile in profiles if profile.region}),
            },
            "model": search_engine.index.manifest["embedding"]["model_id"],
            "profile_count": len(profiles),
            "added_profile_count": len(added),
            "max_added_profiles": search_engine.max_live_additions,
            "synthetic_only": True,
            "local_inference": True,
            "raw_scores_exposed": False,
        })

    @app.post("/api/reset")
    def reset_demo():
        # Discards the in-memory overlay and nothing else.  The sealed artifacts
        # on disk are never written to, so the worst this can do is return the
        # index to the corpus the numbers were measured on — which is why it is
        # left unauthenticated alongside the equally open add endpoint.
        discarded = search_engine.reset()
        if not discarded:
            message = "האינדקס כבר במצבו המקורי"
        elif discarded == 1:
            # Hebrew does not take a plural noun after 1, and "1 פרופילים" reads
            # as a bug to every reader of this UI.
            message = "ההדגמה אופסה — פרופיל אחד שנוסף הוסר מהאינדקס"
        else:
            message = f"ההדגמה אופסה — {discarded} פרופילים שנוספו הוסרו מהאינדקס"
        return jsonify({"status": "ok", "discarded": discarded, "message": message})

    @app.post("/api/search")
    def search():
        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "message": "בקשת החיפוש אינה תקינה"}), 400

        query = payload.get("query")
        filters = payload.get("filters", {})
        if not isinstance(query, str) or not isinstance(filters, dict):
            return jsonify({"status": "error", "message": "בקשת החיפוש אינה תקינה"}), 400

        clean_filters = {
            key: value
            for key, value in filters.items()
            if key in {"sector", "region"} and isinstance(value, str) and value
        }
        try:
            response = search_engine.search(query, filters=clean_filters)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify(response.public_dict())

    @app.get("/api/profiles")
    def list_profiles():
        # A browsable directory of everything a search can currently return —
        # base corpus plus live additions, full narrative fields included. The
        # POC has no authentication, so there is nothing to hide from the demo
        # visitor here; production would put this behind the same Salesforce
        # revalidation and row-level scope as search itself. No scores appear:
        # this is the record, not a ranking.
        # Both halves come from one snapshot: reading the profile list and the
        # added-id set separately lets a concurrent add or reset land between them,
        # and the payload then flags rows against a corpus it is not describing.
        snapshot, added = search_engine.directory_snapshot()
        profiles = [
            {
                "profile_id": profile.profile_id,
                "name": profile.display_name,
                "title": profile.title,
                "organisation": profile.organisation,
                # Contact details are never embedded and never searchable (they
                # sit outside the projection allow-list); they are disclosed
                # here only for a profile the caller is already entitled to see.
                "email": profile.email,
                "phone": profile.phone,
                "sector": profile.sector,
                "region": profile.region,
                "cohort": profile.cohort,
                "description": profile.description,
                "experience": profile.experience,
                "areas_of_activity": profile.areas_of_activity,
                "interests": profile.interests,
                "values": profile.values,
                "affiliations": profile.affiliations,
                "added": profile.profile_id in added,
            }
            for profile in snapshot
        ]
        return jsonify({
            "profiles": profiles,
            "count": len(profiles),
            "added_count": len(added),
        })

    @app.post("/api/profiles")
    def add_profile():
        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "message": "בקשת ההוספה אינה תקינה"}), 400

        def clean(key: str, limit: int = _TEXT_LIMIT) -> str:
            value = payload.get(key, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise ValueError(f"השדה {key} חייב להיות טקסט")
            value = " ".join(value.split())
            if len(value) > limit:
                raise ValueError(f"השדה {key} ארוך מדי (עד {limit} תווים)")
            return value

        try:
            first_name = clean("first_name", _NAME_LIMIT)
            last_name = clean("last_name", _NAME_LIMIT)
            title = clean("title")
            organisation = clean("organisation")
            sector = clean("sector", _NAME_LIMIT)
            region = clean("region", _NAME_LIMIT)
            cohort = clean("cohort", _NAME_LIMIT)
            experience = clean("experience")
            areas = clean("areas_of_activity")
            interests = clean("interests")
            values = clean("values")
            affiliations = clean("affiliations")
            description = clean("description")
            email = clean("email", _NAME_LIMIT)
            phone = clean("phone", _NAME_LIMIT)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        if not first_name or not last_name:
            return jsonify({"status": "error", "message": "נדרשים שם פרטי ושם משפחה"}), 400
        if not any((title, organisation, experience, areas, interests, values, affiliations, description)):
            return jsonify({
                "status": "error",
                "message": "נדרש תוכן באחד משדות התיאור לפחות — אחרת אין מה לחפש",
            }), 400

        field_values = {
            "Title": title,
            "Account.Name": organisation,
            "Sector__c": sector,
            "Region__c": region,
            "Cohort__c": cohort,
            "Experience__c": experience,
            "Areas_of_Activity__c": areas,
            "Interests__c": interests,
            "Values__c": values,
            "Affiliations__c": affiliations,
        }
        # Everything the demo visitor types is treated as staff-entered CRM data
        # (demo_added: trusted on both retrieval legs) — with one deliberate
        # exception. Description maps to the Contact's free-form blurb, which in
        # the sealed corpus is the member's own words, so it keeps the
        # self_described provenance: fully embedded for semantic ranking, excluded
        # from the lexical keyword-rescue, exactly like its corpus counterpart.
        # In production this split comes from Salesforce field history, not a stamp.
        provenance = {field: "demo_added" for field, value in field_values.items() if value}
        if description:
            provenance["Description"] = "self_described"
        record = {
            "_synthetic": True,
            "FirstName": first_name,
            "LastName": last_name,
            "Title": title,
            "Description": description,
            "Account": {"Id": "", "Name": organisation},
            # Contact details never reach the index: the projection allow-list
            # does not name them, so no aspect can carry them into a vector.
            "Email": email,
            "Phone": phone,
            "Sector__c": sector,
            "Region__c": region,
            "Cohort__c": cohort,
            "Areas_of_Activity__c": areas,
            "Experience__c": experience,
            "Interests__c": interests,
            "Values__c": values,
            "Affiliations__c": affiliations,
            "Field_Provenance__c": provenance,
        }
        try:
            profile = search_engine.add_profile(record)
        except (ProfileValidationError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({
            "status": "ok",
            "profile_id": profile.profile_id,
            "name": profile.display_name,
            "message": f"{profile.display_name} נוסף לאינדקס — אפשר לחפש עכשיו",
        }), 201

    return app
