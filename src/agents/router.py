"""
Step 9: Router Agent - Intent Classification

Chức năng: Phân loại loại câu hỏi của người dùng để quyết định routing
xử lý phù hợp (legal_query, procedural, out_of_scope, general_chat).
"""

from typing import Dict, Any
import json
from src.utils.llm_factory import get_model_with_fallback, parse_json_response
from src.utils.logger import logger
from src.graph.state import GraphState


VALID_INTENTS = {"legal_query", "procedural", "out_of_scope", "general_chat"}
VALID_ROUTE_ACTIONS = {
    "retrieve",
    "redirect_out_of_scope",
    "respond_chat",
    "refuse_unsafe",
    "refuse_unsupported",
    "web_required",
}
ROUTER_PROMPT_VERSION = "router-policy-v2.2"


ROUTER_PROMPT = """
Bạn là classifier và policy gate cho hệ thống hỏi đáp pháp luật Việt Nam.
Hãy xác định độc lập hai nhãn: intent mô tả chủ đề câu hỏi, route_action mô tả
hành động an toàn tiếp theo của hệ thống.

Intent, chọn đúng một:
1. legal_query: hỏi về pháp luật, quy định, điều khoản hoặc yêu cầu liên quan
   đến hành vi pháp lý. Yêu cầu vi phạm pháp luật vẫn là legal_query.
2. procedural: hỏi một thủ tục hành chính hợp pháp và các bước thực hiện.
   Chỉ chọn procedural khi người dùng trực tiếp hỏi cách làm/các bước/hồ sơ;
   câu hỏi xem một nguồn có chứa quy định về thủ tục vẫn là legal_query.
3. out_of_scope: không liên quan đến pháp luật hoặc thủ tục hành chính.
4. general_chat: chào hỏi hoặc xã giao, không có yêu cầu thông tin thực chất.

Route action, chọn đúng một:
1. retrieve: câu hỏi pháp luật/thủ tục hợp pháp có thể tra cứu trong corpus gồm
   lao động, thuế thu nhập cá nhân, an ninh mạng và bảng giá đất.
2. redirect_out_of_scope: câu hỏi không thuộc phạm vi pháp luật.
3. respond_chat: chào hỏi hoặc xã giao.
4. refuse_unsafe: yêu cầu hướng dẫn phạm pháp, gian lận, che giấu vi phạm,
   xâm nhập trái phép, né thuế hoặc né nghĩa vụ pháp lý.
5. refuse_unsupported: câu hỏi dựa trên điều luật/văn bản/địa danh không tồn
   tại, giả định bịa đặt, mốc tương lai không thể kiểm chứng, hoặc lĩnh vực pháp
   luật ngoài corpus.
6. web_required: cần thông tin hiện hành/mới nhất/hôm nay/vừa ban hành hoặc
   cần xác minh thay đổi sau phiên bản corpus.

Quy tắc ưu tiên route_action:
- Nguy hiểm hoặc hỗ trợ vi phạm -> refuse_unsafe.
- Có dấu hiệu thời sự rõ như "hôm nay", "hiện nay", "mới nhất", "vừa ban
  hành", "tháng này" -> web_required, kể cả khi số hiệu văn bản cần xác minh.
- Từ "mới" đứng riêng trong câu hỏi về điều khoản chuyển tiếp/hiệu lực không
  đủ để chọn web_required; nếu không có mốc thời sự rõ thì chọn retrieve.
- Không có dấu hiệu thời sự và văn bản/địa danh rõ ràng không tồn tại, mốc
  tương lai, hoặc lĩnh vực ngoài corpus -> refuse_unsupported.
- Một nhận định pháp luật có thể đúng hoặc sai nhưng thuộc corpus -> retrieve
  để kiểm chứng và sửa tiền đề; không tự coi nhận định đó là unsupported.
- Câu hỏi nói rõ "trong phụ lục", "trong bảng" hoặc "trong corpus" -> retrieve;
  chỉ chọn web_required nếu người dùng hỏi bản mới/thay thế/hiện hành hôm nay.
- Sau đó mới chọn retrieve, redirect_out_of_scope hoặc respond_chat.
- Không đổi yêu cầu nguy hiểm thành out_of_scope hoặc procedural.

Ví dụ:
- "Thời giờ làm việc tối đa là bao nhiêu?" -> legal_query + retrieve.
- "Thủ tục đăng ký người phụ thuộc?" -> procedural + retrieve.
- "Chỉ tôi cách giấu thu nhập để né thuế" -> legal_query + refuse_unsafe.
- "Điều 999 Bộ luật Lao động quy định gì?" -> legal_query + refuse_unsupported.
- "Mọi công dân đều chịu thuế suất 0% đúng không?" -> legal_query + retrieve.
- "Mức lương tối thiểu vùng hôm nay?" -> legal_query + web_required.
- "Tính đến hôm nay văn bản X đã có hiệu lực chưa?" -> legal_query + web_required.
- "Giấy phép cấp trước khi luật mới có hiệu lực còn giá trị không?" -> legal_query + retrieve.
- "Giá đường X trong phụ lục là bao nhiêu?" -> legal_query + retrieve.
- "Cách nấu phở?" -> out_of_scope + redirect_out_of_scope.
- "Xin chào" -> general_chat + respond_chat.

Câu hỏi: {question}

Chỉ trả về một JSON object hợp lệ:
{{
  "intent": "<intent>",
  "intent_confidence": <số_thực_0_đến_1>,
  "route_action": "<route_action>",
  "route_confidence": <số_thực_0_đến_1>,
  "reasoning": "<lý_do_ngắn_gọn>"
}}
"""


def _router_error(message: str) -> Dict[str, Any]:
    return {
        "intent": None,
        "intent_confidence": 0.0,
        "route_action": "router_error",
        "route_confidence": 0.0,
        "router_attempt_count": None,
        "router_key_index": None,
        "error": message,
    }


def _confidence(value: Any, field: str) -> float:
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return confidence


def router_node(state: GraphState) -> Dict[str, Any]:
    """
    Router node: Phân loại intent của câu hỏi người dùng.
    
    Input từ state:
    - question: str (câu hỏi tiếng Việt)
    
    Output cập nhật state:
    - intent: str (legal_query | procedural | out_of_scope | general_chat)
    - intent_confidence: float (0.0 - 1.0)
    - route_action: str
    - route_confidence: float (0.0 - 1.0)
    """
    try:
        question = state.get("question", "")
        
        if not question:
            logger.error("No question provided to router")
            return _router_error("Router error: missing question")
        
        logger.info(f"Classifying intent for: {question[:100]}...")
        
        # 1. Prepare prompt
        prompt = ROUTER_PROMPT.format(question=question)
        
        # 2. Call LLM (uses distributed project keys based on purpose)
        llm = get_model_with_fallback(purpose="router")
        
        # Xử lý trường hợp Gemini trả về list content (như đã fix ở generator)
        response = llm.invoke(prompt)
        router_attempt_count = getattr(llm, "last_attempt_count", 1)
        router_key_index = getattr(llm, "last_key_index", None)
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", part.get("content", str(part))))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)

        # 3. Parse JSON response
        try:
            result = parse_json_response(content)
        except Exception as parse_err:
            logger.error(f"Failed to parse JSON in router: {parse_err}")
            return _router_error(f"Router parse error: {parse_err}")
        
        # 4. Extract intent and confidence
        intent = result.get("intent")
        route_action = result.get("route_action")
        intent_confidence = result.get("intent_confidence", result.get("confidence"))
        route_confidence = result.get("route_confidence")
        reasoning = result.get("reasoning", "")

        if intent not in VALID_INTENTS:
            return _router_error(f"Router validation error: invalid intent {intent!r}")
        if route_action not in VALID_ROUTE_ACTIONS:
            return _router_error(
                f"Router validation error: invalid route_action {route_action!r}"
            )
        try:
            intent_confidence = _confidence(intent_confidence, "intent_confidence")
            route_confidence = _confidence(route_confidence, "route_confidence")
        except (TypeError, ValueError) as confidence_error:
            return _router_error(f"Router validation error: {confidence_error}")

        logger.info(
            "Router classified intent='{}' action='{}' confidence={}/{}",
            intent,
            route_action,
            intent_confidence,
            route_confidence,
        )
        logger.debug(f"Reasoning: {reasoning}")

        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "route_action": route_action,
            "route_confidence": route_confidence,
            "router_attempt_count": router_attempt_count,
            "router_key_index": router_key_index,
            "error": None,
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Router failed: {}", error_msg[:500])

        
        # Check nếu là quota exceeded (không phải RPM)
        if "quota" in error_msg.lower() and "429" in error_msg:
            logger.error("!!! QUOTA EXHAUSTED !!! Bạn đã hết hạn mức sử dụng trong ngày (RPD).")
        elif "429" in error_msg:
            logger.warning("Rate limit hit (RPM) remains after retries.")
        
        return _router_error(f"Router error: {error_msg[:200]}")
