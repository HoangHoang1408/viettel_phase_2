import requests

HEADERS = {
    "Authorization": "Bearer eyJhY3RpdmUiOjEsInJldmlld2VyIjowLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJkYXRodDE3IiwiZXhwIjoxNjkyMjQxOTAzLCJzY29wZSI6W119.__pqpWjeAkEwv5UOKEOUPWNAXrTJ4RhhEZit4LpL3LusFMwZM1cMrs_GLqGowunMBKtAKe7bWx-cMDHC7btbOQ",
    "Device-UUID": "b8d4bcf4-68be-4e31-9946-c684b56b5778",
    "type": "tla-url-web-search",
}
SEARCH_REUQEST_HOST = (
    "https://gateway-tla.toaan.gov.vn/mobile/search/text/611f8300e6b7d5f40ee6f9bd"
)
LAW_DETAIL_REQUEST_HOST = (
    "https://gateway-tla.toaan.gov.vn/api/chatbot/lawDetailForSearchAll?id={id_}"
)


class Retriever:
    def __init__(
        self,
        request_headers=HEADERS,
        search_request_host=SEARCH_REUQEST_HOST,
        law_detail_request_host=LAW_DETAIL_REQUEST_HOST,
    ):
        self.request_headers = request_headers
        self.search_request_host = search_request_host
        self.law_detail_request_host = law_detail_request_host

    def retrieve(self, query, take=3, only_semantic=False):
        results = []
        try:
            res = requests.post(
                self.search_request_host,
                headers=self.request_headers,
                json={"word": query, "typeSearch": "law_detail"},
            ).json()

            semantic_results, non_semantic_results = [], []
            for law in res["data"]:
                if "metadatas" not in law:
                    continue
                for article in law["metadatas"]:
                    if article["resultType"] == "SEMANTIC_SEARCH":
                        semantic_results.append(
                            {
                                "id": article["id"],
                            }
                        )
                    else:
                        non_semantic_results.append(
                            {
                                "id": article["id"],
                            }
                        )

            if only_semantic:
                results.extend(semantic_results)
            else:
                results.extend(semantic_results + non_semantic_results)
            results = results[:take]

            for i, result in enumerate(results):
                res = requests.get(
                    self.law_detail_request_host.format(id_=result["id"]),
                    headers=self.request_headers,
                ).json()
                result["title"] = res["data"]["nameDisplay"].replace("\n", ". ")
                result["content"] = res["data"]["content"]
                result["score"] = 1
            results = sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            print("###FOCUS### Error when retrieving")
            print(e)
            return []
        return [t["title"] + "\n" + t["content"] for t in results]
