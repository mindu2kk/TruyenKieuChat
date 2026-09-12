(function () {
  const SUGGESTION_SETS = [
    [
      { icon: "✦", title: "Bình giảng đoạn thơ", prompt: "Bình giảng câu “Cậy em em có chịu lời” và đặt câu thơ vào bối cảnh đoạn Trao duyên." },
      { icon: "人", title: "Khám phá nhân vật", prompt: "Vì sao Thúy Kiều quyết định bán mình chuộc cha?" },
      { icon: "❝", title: "Tìm câu thơ", prompt: "Tìm những câu thơ trong Truyện Kiều nói về nỗi nhớ." },
      { icon: "⌁", title: "Theo dòng truyện", prompt: "Điều gì xảy ra sau cuộc gặp gỡ đầu tiên giữa Thúy Kiều và Kim Trọng?" },
      { icon: "冊", title: "Giải nghĩa điển tích", prompt: "“Quạt nồng ấp lạnh” nghĩa là gì trong Truyện Kiều?" },
      { icon: "✧", title: "Phân tích nghệ thuật", prompt: "Nguyễn Du miêu tả nội tâm Thúy Kiều bằng những thủ pháp nào?" }
    ],
    [
      { icon: "✦", title: "Đọc một đoạn thơ", prompt: "Trích chính xác 10 câu mở đầu Truyện Kiều và giải thích ý nghĩa." },
      { icon: "人", title: "Chân dung Từ Hải", prompt: "Phân tích vẻ đẹp lý tưởng của nhân vật Từ Hải." },
      { icon: "❝", title: "Tìm lời tự tình", prompt: "Tìm những câu thơ Thúy Kiều tự nói về thân phận mình." },
      { icon: "⌁", title: "Đoạn Trao duyên", prompt: "Tóm tắt diễn biến tâm lý Thúy Kiều trong đoạn Trao duyên." },
      { icon: "冊", title: "Giải nghĩa từ cổ", prompt: "Giải nghĩa các từ cổ quan trọng trong đoạn Chị em Thúy Kiều." },
      { icon: "✧", title: "So sánh hình tượng", prompt: "So sánh cách Nguyễn Du miêu tả Thúy Vân và Thúy Kiều." }
    ],
    [
      { icon: "✦", title: "Bình một chữ", prompt: "Phân tích giá trị của chữ “tâm” trong hai câu kết Truyện Kiều." },
      { icon: "人", title: "Hiểu Kim Trọng", prompt: "Kim Trọng là nhân vật như thế nào trong Truyện Kiều?" },
      { icon: "❝", title: "Tìm cảnh ngụ tình", prompt: "Tìm và phân tích một đoạn thơ tiêu biểu cho bút pháp tả cảnh ngụ tình." },
      { icon: "⌁", title: "Kiều ở lầu Ngưng Bích", prompt: "Chuyện gì dẫn đến việc Thúy Kiều ở lầu Ngưng Bích?" },
      { icon: "冊", title: "Hiểu một thành ngữ", prompt: "“Bên tình bên hiếu bên nào nặng hơn” thể hiện bi kịch gì?" },
      { icon: "✧", title: "Nghệ thuật kể chuyện", prompt: "Phân tích nghệ thuật xây dựng tình huống trong Truyện Kiều." }
    ]
  ];

  const DAILY_VERSES = [
    { "lineStart": 1243, "lineEnd": 1244, "lines": "Cảnh nào cảnh chẳng đeo sầu?\nNgười buồn, cảnh có vui đâu bao giờ!", "prompt": "Bình giảng hai câu 1243-1244 trong Truyện Kiều, tập trung vào bút pháp tả cảnh ngụ tình." },
    { "lineStart": 3251, "lineEnd": 3252, "lines": "Thiện căn ở tại lòng ta,\nChữ tâm kia mới bằng ba chữ tài!", "prompt": "Bình giảng hai câu 3251-3252 trong Truyện Kiều, tập trung vào quan niệm về chữ tâm." },
    { "lineStart": 1, "lineEnd": 2, "lines": "Trăm năm, trong cõi người ta,\nChữ tài, chữ mệnh, khéo là ghét nhau.", "prompt": "Bình giảng hai câu 1-2 mở đầu Truyện Kiều." },
    { "lineStart": 1795, "lineEnd": 1796, "lines": "Sen tàn, cúc lại nở hoa,\nSầu dài, ngày ngắn, đông đà sang xuân.", "prompt": "Bình giảng hai câu 1795-1796 trong Truyện Kiều, tập trung vào cảm thức thời gian." },
    { "lineStart": 163, "lineEnd": 164, "lines": "Người quốc sắc, kẻ thiên tài,\nTình trong như đã, mặt ngoài còn e.", "prompt": "Bình giảng hai câu 163-164 trong Truyện Kiều về cuộc gặp Thúy Kiều và Kim Trọng." }
  ];

  const LOADING_MESSAGES = [
    "Kiều Bot đang tra văn bản…",
    "Kiều Bot đang đọc lại đoạn thơ…",
    "Kiều Bot đang nối bối cảnh và ý nghĩa…"
  ];

  function cleanTemplateValue(value, fallback) {
    const text = String(value || "").trim();
    return !text || text.includes("{{") || text.includes("{%") ? fallback : text;
  }

  function getConfig() {
    const root = document.getElementById("kieu-app");
    const authenticated = cleanTemplateValue(root?.dataset.authenticated, "false") === "true";
    return {
      authenticated,
      userName: cleanTemplateValue(root?.dataset.userName, authenticated ? "Bạn đọc" : "Khách"),
      userInitial: cleanTemplateValue(root?.dataset.userInitial, authenticated ? "B" : "↗"),
      chatUrl: cleanTemplateValue(root?.dataset.chatUrl, "/api/chat/"),
      historyUrl: cleanTemplateValue(root?.dataset.historyUrl, "/api/history/"),
      conversationsUrl: cleanTemplateValue(root?.dataset.conversationsUrl, "/api/conversations/"),
      conversationDetailBase: cleanTemplateValue(root?.dataset.conversationDetailBase, "/api/conversations/"),
      messageActionsBase: cleanTemplateValue(root?.dataset.messageActionsBase, "/api/messages/"),
      loginUrl: cleanTemplateValue(root?.dataset.loginUrl, "/accounts/login/"),
      logoutUrl: cleanTemplateValue(root?.dataset.logoutUrl, "/accounts/logout/"),
      model: cleanTemplateValue(root?.dataset.model, "gemini-2.5-flash"),
      geminiReady: cleanTemplateValue(root?.dataset.geminiReady, "true") === "true",
      poemReady: cleanTemplateValue(root?.dataset.poemReady, "true") === "true"
    };
  }

  window.ChatUI = function ChatUI() {
    return {
      typing: false,
      sending: false,
      draft: "",
      sidebarOpen: false,
      sidebarCollapsed: false,
      compactNav: false,
      historyLoaded: false,
      hasMessages: false,
      showJumpToLatest: false,
      recentMenuOpen: false,
      recentTitle: "Mạch đọc hiện tại",
      conversations: [],
      currentConversationId: "legacy",
      notice: "",
      liveStatus: "",
      loadingLabel: LOADING_MESSAGES[0],
      theme: "ink",
      suggestions: SUGGESTION_SETS[0],
      dailyVerse: DAILY_VERSES[0],
      config: getConfig(),
      settings: { response_length: "short", debug_meta: false },

      init() {
        this.restorePreferences();
        this.pickDiscoverySet();
        this.pickDailyVerse();
        this.syncViewport();
        document.documentElement.dataset.theme = this.theme === "paper" ? "paper" : "ink";
        window.addEventListener("resize", () => this.syncViewport());
        window.addEventListener("scroll", () => this.updateScrollState(), { passive: true });
        window.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            this.recentMenuOpen = false;
            this.closeSidebar(true);
          }
          if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
            event.preventDefault();
            this.focusSearch();
          }
        });
        this.bootstrapConversations();
        this.$watch("settings", () => this.savePreferences(), { deep: true });
      },

      restorePreferences() {
        try {
          const saved = JSON.parse(localStorage.getItem("kieu.chat-preferences.v2") || "{}");
          this.settings = { ...this.settings, ...(saved.settings || {}) };
          this.sidebarCollapsed = Boolean(saved.sidebarCollapsed);
          this.theme = saved.theme === "paper" ? "paper" : "ink";
          this.recentTitle = String(saved.recentTitle || this.recentTitle).slice(0, 64);
        } catch (error) {
          localStorage.removeItem("kieu.chat-preferences.v2");
        }
        if (!["super_short", "short", "long"].includes(this.settings.response_length)) this.settings.response_length = "short";
        this.settings.debug_meta = false;
      },

      savePreferences() {
        localStorage.setItem("kieu.chat-preferences.v2", JSON.stringify({
          settings: this.settings,
          sidebarCollapsed: this.sidebarCollapsed,
          theme: this.theme,
          recentTitle: this.recentTitle,
          currentConversationId: this.currentConversationId
        }));
      },

      csrfToken() {
        return document.querySelector('meta[name="csrf-token"]')?.content || "";
      },

      async bootstrapConversations() {
        try {
          const response = await fetch(this.config.conversationsUrl, { credentials: "same-origin" });
          if (!response.ok) throw new Error("conversation list unavailable");
          const data = await response.json();
          this.conversations = data.conversations || [];
          const savedId = this.getSavedConversationId();
          this.currentConversationId = this.conversations.some((item) => item.id === savedId)
            ? savedId
            : (this.conversations[0]?.id || "legacy");
          const current = this.conversations.find((item) => item.id === this.currentConversationId);
          if (current) this.recentTitle = current.title;
        } catch (error) {
          console.error(error);
          this.currentConversationId = "legacy";
        }
        await this.reloadHistory();
      },

      getSavedConversationId() {
        try {
          return JSON.parse(localStorage.getItem("kieu.chat-preferences.v2") || "{}").currentConversationId || "legacy";
        } catch (error) {
          return "legacy";
        }
      },

      pickDiscoverySet() {
        const last = Number(sessionStorage.getItem("kieu.suggestion-set") || -1);
        let next = Math.floor(Math.random() * SUGGESTION_SETS.length);
        if (next === last) next = (next + 1) % SUGGESTION_SETS.length;
        sessionStorage.setItem("kieu.suggestion-set", String(next));
        this.suggestions = SUGGESTION_SETS[next];
      },

      pickDailyVerse() {
        const day = Math.floor(Date.now() / 86400000);
        this.dailyVerse = DAILY_VERSES[day % DAILY_VERSES.length];
      },

      syncViewport() {
        this.compactNav = window.matchMedia("(max-width: 820px)").matches;
        if (!this.compactNav) this.sidebarOpen = false;
      },

      toggleSidebar() {
        if (this.compactNav) this.sidebarOpen = !this.sidebarOpen;
        else {
          this.sidebarCollapsed = !this.sidebarCollapsed;
          this.savePreferences();
        }
      },

      closeSidebar(restoreFocus) {
        if (!this.compactNav || !this.sidebarOpen) return;
        this.sidebarOpen = false;
        if (restoreFocus) this.$nextTick(() => this.$refs.menu?.focus());
      },

      toggleTheme() {
        this.theme = this.theme === "ink" ? "paper" : "ink";
        document.documentElement.dataset.theme = this.theme === "paper" ? "paper" : "ink";
        this.savePreferences();
        this.flash(this.theme === "paper" ? "Đã bật nền giấy dịu." : "Đã trở về nền mực.");
      },

      focusSearch() {
        this.draft = "Tìm câu thơ trong Truyện Kiều về ";
        this.$nextTick(() => {
          this.$refs.composer?.focus();
          this.resizeComposer({ target: this.$refs.composer });
        });
      },

      flash(message) {
        this.notice = message;
        window.clearTimeout(this.noticeTimer);
        this.noticeTimer = window.setTimeout(() => { this.notice = ""; }, 3000);
      },

      resizeComposer(event) {
        const textarea = event.target;
        if (!textarea) return;
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(168, textarea.scrollHeight)}px`;
      },

      updateScrollState() {
        const pageHeight = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
        this.showJumpToLatest = this.hasMessages && pageHeight - window.scrollY - window.innerHeight > 180;
      },

      async reloadHistory() {
        try {
          const historyUrl = `${this.config.historyUrl}?conversation_id=${encodeURIComponent(this.currentConversationId)}`;
          const response = await fetch(historyUrl, { credentials: "same-origin" });
          if (!response.ok) throw new Error("history unavailable");
          const data = await response.json();
          const messages = data.messages || [];
          const box = document.getElementById("chat");
          box.innerHTML = "";
          const firstQuestion = messages.find((message) => message.role === "user")?.content;
          if (firstQuestion && this.recentTitle === "Mạch đọc hiện tại") this.recentTitle = this.topicFrom(firstQuestion);
          messages.forEach((message) => this.render(message.role, message.content, message.meta || null, {
            messageId: message.id,
            actions: message.actions || {}
          }));
          this.hasMessages = messages.length > 0;
          this.addFollowupsToLast();
          this.liveStatus = this.hasMessages ? `Đã tải ${messages.length} tin nhắn trong mạch đọc.` : "Phòng đọc đã sẵn sàng.";
          this.scrollBottom(true);
        } catch (error) {
          console.error(error);
          this.flash("Không thể tải mạch đọc gần đây.");
          this.liveStatus = "Không thể tải mạch đọc gần đây.";
        } finally {
          this.historyLoaded = true;
        }
      },

      async selectConversation(conversationId) {
        if (conversationId === this.currentConversationId) {
          this.closeSidebar(false);
          this.$refs.composer?.focus();
          return;
        }
        this.currentConversationId = conversationId;
        this.recentMenuOpen = false;
        const current = this.conversations.find((item) => item.id === conversationId);
        this.recentTitle = current?.title || "Mạch đọc";
        this.historyLoaded = false;
        this.closeSidebar(false);
        this.savePreferences();
        await this.reloadHistory();
      },

      topicFrom(message) {
        const compact = String(message || "").replace(/\s+/g, " ").trim();
        return compact.length > 42 ? `${compact.slice(0, 42).trim()}…` : compact || "Mạch đọc hiện tại";
      },

      async renameConversation(conversation) {
        this.recentMenuOpen = false;
        const value = window.prompt("Đặt tên cho mạch đọc", conversation.title);
        if (!value?.trim()) return;
        const title = value.trim().slice(0, 80);
        try {
          const response = await fetch(`${this.config.conversationDetailBase}${encodeURIComponent(conversation.id)}/`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken() },
            credentials: "same-origin",
            body: JSON.stringify({ title })
          });
          if (!response.ok) throw new Error("rename failed");
          conversation.title = title;
          if (conversation.id === this.currentConversationId) this.recentTitle = title;
          this.savePreferences();
          this.flash("Đã đổi tên mạch đọc.");
        } catch (error) {
          this.flash("Chưa thể đổi tên mạch đọc.");
        }
      },

      async newConversation() {
        try {
          const response = await fetch(this.config.conversationsUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken() },
            credentials: "same-origin",
            body: JSON.stringify({ title: "Cuộc trò chuyện mới" })
          });
          if (!response.ok) throw new Error("create failed");
          const data = await response.json();
          this.conversations.unshift(data.conversation);
          this.currentConversationId = data.conversation.id;
          this.recentTitle = data.conversation.title;
          this.hasMessages = false;
          this.historyLoaded = true;
          document.getElementById("chat").innerHTML = "";
          this.pickDiscoverySet();
          this.closeSidebar(false);
          this.savePreferences();
          this.$nextTick(() => this.$refs.composer?.focus());
        } catch (error) {
          this.flash("Chưa thể tạo mạch đọc mới.");
        }
      },

      async deleteConversationById(conversationId) {
        this.recentMenuOpen = false;
        if (!window.confirm("Bạn có chắc muốn xóa mạch đọc này?")) return;
        try {
          const response = await fetch(`${this.config.conversationDetailBase}${encodeURIComponent(conversationId)}/`, {
            method: "DELETE",
            headers: { "X-CSRFToken": this.csrfToken() },
            credentials: "same-origin"
          });
          if (!response.ok) throw new Error("delete failed");
          this.conversations = this.conversations.filter((item) => item.id !== conversationId);
          if (conversationId === this.currentConversationId) {
            this.currentConversationId = this.conversations[0]?.id || "legacy";
            this.recentTitle = this.conversations[0]?.title || "Mạch đọc hiện tại";
            await this.reloadHistory();
          }
          this.savePreferences();
          this.flash("Đã xóa mạch đọc.");
        } catch (error) {
          this.flash("Chưa thể xóa mạch đọc.");
        }
      },

      clearChat(fromNewConversation) {
        const action = fromNewConversation ? "bắt đầu một cuộc trò chuyện mới" : "xóa toàn bộ mạch đọc này";
        if (!window.confirm(`Bạn có chắc muốn ${action}?`)) return;
        fetch(`${this.config.historyUrl}?conversation_id=${encodeURIComponent(this.currentConversationId)}`, {
          method: "DELETE",
          headers: { "X-CSRFToken": this.csrfToken() },
          credentials: "same-origin"
        })
          .then((response) => {
            if (!response.ok) throw new Error("Unable to clear history");
            this.conversations = this.conversations.filter((item) => item.id !== this.currentConversationId);
            this.currentConversationId = this.conversations[0]?.id || "legacy";
            this.recentTitle = this.conversations[0]?.title || "Mạch đọc hiện tại";
            this.pickDiscoverySet();
            this.savePreferences();
            return this.reloadHistory();
          })
          .then(() => {
            this.closeSidebar(false);
            this.flash("Mạch đọc đã được làm mới.");
            this.$refs.composer?.focus();
          })
          .catch(() => this.flash("Chưa thể xóa mạch đọc. Hãy thử lại."));
      },

      ask(prompt, sendNow = true) {
        this.draft = prompt;
        this.closeSidebar(false);
        this.$nextTick(() => {
          this.resizeComposer({ target: this.$refs.composer });
          if (sendNow) this.send();
          else this.$refs.composer?.focus();
        });
      },

      presentAssistantContent(content) {
        const value = String(content || "");
        return /Gọi Gemini thất bại|RESOURCE_EXHAUSTED|\bUNAVAILABLE\b|Chi tiết kỹ thuật:/i.test(value)
          ? "Kiều Bot đang gặp một trục trặc tạm thời khi tra cứu. Mạch đọc của bạn vẫn được giữ nguyên — hãy thử gửi lại câu hỏi sau ít phút."
          : value;
      },

      render(role, content, meta, options = {}) {
        const box = document.getElementById("chat");
        const template = document.getElementById(role === "user" ? "tpl-user" : "tpl-bot");
        const fragment = template.content.cloneNode(true);
        const message = fragment.querySelector(".msg");
        if (role === "assistant") {
          const displayContent = this.presentAssistantContent(content);
          const answerId = options.messageId || this.answerId(displayContent);
          message.innerHTML = Array.isArray(meta?.content_blocks) && meta.content_blocks.length
            ? this.renderBlocks(meta.content_blocks)
            : this.markdown(displayContent);
          const article = fragment.querySelector("article");
          article.dataset.answerId = answerId;
          fragment.querySelector('[data-copy-answer]')?.addEventListener("click", () => this.copyAnswer(displayContent));
          fragment.querySelector('[data-save-answer]')?.addEventListener("click", (event) => this.toggleSave(event.currentTarget, answerId, displayContent));
          fragment.querySelector('[data-feedback-up]')?.addEventListener("click", (event) => this.setFeedback(event.currentTarget, answerId, "up"));
          fragment.querySelector('[data-feedback-down]')?.addEventListener("click", (event) => this.setFeedback(event.currentTarget, answerId, "down"));
          fragment.querySelector('[data-deepen]')?.addEventListener("click", () => this.ask("Hãy đào sâu thêm câu trả lời vừa rồi, tập trung vào bối cảnh và nghệ thuật của Nguyễn Du."));
          this.restoreActionState(fragment, answerId, options.actions || {});
          this.renderMetadata(fragment, meta);
        } else {
          message.textContent = content;
        }
        box.appendChild(fragment);
        this.hasMessages = true;
      },

      answerId(content) {
        let hash = 0;
        for (let index = 0; index < content.length; index += 1) hash = ((hash << 5) - hash + content.charCodeAt(index)) | 0;
        return `answer-${Math.abs(hash)}`;
      },

      getLocalMap(key) {
        try { return JSON.parse(localStorage.getItem(key) || "{}"); }
        catch (error) { return {}; }
      },

      restoreActionState(fragment, answerId, serverActions = {}) {
        const saved = this.getLocalMap("kieu.saved-answers.v1");
        const feedback = this.getLocalMap("kieu.answer-feedback.v1");
        const isSaved = serverActions.saved ?? Boolean(saved[answerId]);
        const feedbackValue = serverActions.feedback || feedback[answerId];
        fragment.querySelector('[data-save-answer]')?.classList.toggle("is-active", isSaved);
        fragment.querySelector(`[data-feedback-${feedbackValue}]`)?.classList.add("is-active");
      },

      async toggleSave(button, answerId, content) {
        const saved = this.getLocalMap("kieu.saved-answers.v1");
        const nextSaved = !button.classList.contains("is-active");
        if (!nextSaved) delete saved[answerId];
        else saved[answerId] = { content, savedAt: new Date().toISOString() };
        localStorage.setItem("kieu.saved-answers.v1", JSON.stringify(saved));
        button.classList.toggle("is-active", nextSaved);
        await this.syncMessageAction(answerId, { saved: nextSaved });
        this.flash(nextSaved ? "Đã lưu câu trả lời." : "Đã bỏ lưu câu trả lời.");
      },

      async setFeedback(button, answerId, value) {
        const feedback = this.getLocalMap("kieu.answer-feedback.v1");
        const article = button.closest("article");
        const wasActive = button.classList.contains("is-active");
        article.querySelectorAll("[data-feedback-up], [data-feedback-down]").forEach((item) => item.classList.remove("is-active"));
        let nextFeedback = value;
        if (wasActive) {
          delete feedback[answerId];
          nextFeedback = null;
        }
        else {
          feedback[answerId] = value;
          button.classList.add("is-active");
        }
        localStorage.setItem("kieu.answer-feedback.v1", JSON.stringify(feedback));
        await this.syncMessageAction(answerId, { feedback: nextFeedback });
        this.flash(nextFeedback ? "Cảm ơn bạn đã góp ý." : "Đã bỏ đánh giá.");
      },

      async syncMessageAction(messageId, payload) {
        if (!/^[a-f0-9]{24}$/i.test(String(messageId))) return;
        try {
          const response = await fetch(`${this.config.messageActionsBase}${encodeURIComponent(messageId)}/actions/`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken() },
            credentials: "same-origin",
            body: JSON.stringify(payload)
          });
          if (!response.ok) throw new Error("action sync failed");
        } catch (error) {
          this.flash("Thao tác đã lưu cục bộ; chưa thể đồng bộ máy chủ.");
        }
      },

      renderBlocks(blocks) {
        return blocks.map((block) => {
          if (block.type === "verse_quote") {
            const line = block.line_start ? `<span class="block-meta">Câu ${block.line_start}${block.line_end && block.line_end !== block.line_start ? `–${block.line_end}` : ""}</span>` : "";
            return `<blockquote class="verse-quote">${this.inline(block.text || "")}${line}</blockquote>`;
          }
          if (block.type === "character_card") {
            const traits = (block.traits || []).map((trait) => `<span>${this.escape(trait)}</span>`).join("");
            return `<aside class="character-card"><span class="block-kicker">Hồ sơ nhân vật</span><h3>${this.escape(block.name)}</h3><p><b>Vai trò</b>${this.escape(block.role || "")}</p><div class="trait-list">${traits}</div></aside>`;
          }
          if (block.type === "timeline") {
            const items = (block.items || []).map((item, index) => `<li class="${index === 0 ? "is-current" : ""}"><span></span>${this.inline(item)}</li>`).join("");
            return `<section class="timeline-block"><span class="block-kicker">${this.escape(block.title || "Dòng sự kiện")}</span><ol>${items}</ol></section>`;
          }
          if (block.type === "context") {
            return `<section class="context-block"><span class="block-kicker">${this.escape(block.title || "Bối cảnh")}</span>${this.markdown(block.markdown || "")}</section>`;
          }
          if (block.type === "reference") {
            return `<div class="reference-block"><span>✓ ${this.escape(block.label || "Kiểm chứng văn bản")}</span><span>${Math.round((block.coverage || 0) * 100)}% · ${Number(block.quote_count || 0)} trích dẫn</span></div>`;
          }
          const className = block.type === "summary" ? "summary-block" : "analysis-block";
          return `<section class="${className}">${this.markdown(block.markdown || "")}</section>`;
        }).join("");
      },

      async copyAnswer(content) {
        try {
          await navigator.clipboard.writeText(content);
          this.flash("Đã sao chép câu trả lời.");
        } catch (error) {
          this.flash("Trình duyệt chưa cho phép sao chép tự động.");
        }
      },

      renderMetadata(fragment, meta) {
        if (!meta) return;
        const summary = fragment.querySelector('[data-meta-summary]');
        const detail = fragment.querySelector('[data-meta-details]');
        const harness = meta.harness || {};
        const quality = harness.quality || {};
        summary.hidden = !this.settings.debug_meta;
        fragment.querySelector('[data-meta-intent]').textContent = `🧭 ${meta.intent || "domain"} · ${harness.flow || "default"}`;
        fragment.querySelector('[data-meta-elapsed]').textContent = `⏱ ${Math.round(meta.elapsed_ms || 0)} ms`;
        const budget = harness.token_budget;
        fragment.querySelector('[data-meta-budget]').textContent = !budget ? "🧠 legacy" : budget.max_output_tokens ? `🧠 ${budget.max_output_tokens} tokens · ${budget.tier || "auto"}` : "🧠 deterministic";
        fragment.querySelector('[data-meta-quality]').textContent = `🛡 ${quality.status || "unchecked"}`;
        detail.hidden = !this.settings.debug_meta;
        detail.innerHTML = this.renderMeta(meta);
      },

      renderMeta(meta) {
        let html = "";
        const harness = meta.harness || {};
        const quality = harness.quality || {};
        if (harness.route_reason) html += `<div>Route: ${this.escape(harness.route_reason)} · confidence ${Math.round((harness.route_confidence || 0) * 100)}%</div>`;
        if (harness.token_budget && Array.isArray(harness.token_budget.reasons)) html += `<div>Token plan: ${this.escape(harness.token_budget.reasons.join(", "))}</div>`;
        if (Array.isArray(quality.issues) && quality.issues.length) html += `<div>${this.escape(quality.issues.join(", "))}</div>`;
        if (meta.verification && meta.verification.coverage !== undefined) html += `<div>Kiểm chứng trích dẫn: coverage ~${Math.round((meta.verification.coverage || 0) * 100)}%</div>`;
        return html;
      },

      addFollowupsToLast() {
        document.querySelectorAll(".followups").forEach((node) => { node.innerHTML = ""; });
        const article = document.querySelector("#chat .chat-message.assistant:last-of-type");
        if (!article) return;
        const area = article.querySelector(".followups");
        ["Phân tích nghệ thuật", "Xem bối cảnh", "Tâm lý Thúy Kiều"].forEach((label) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "followup-chip";
          button.textContent = label;
          button.addEventListener("click", () => this.ask(`${label} sâu hơn dựa trên câu trả lời vừa rồi.`));
          area.appendChild(button);
        });
      },

      escape(value) {
        return String(value || "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
      },

      inline(text) {
        return this.escape(text)
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/\*(.+?)\*/g, "<em>$1</em>")
          .replace(/`(.+?)`/g, "<code>$1</code>");
      },

      markdown(value) {
        const lines = String(value || "").replace(/\r/g, "").split("\n");
        const html = [];
        let paragraph = [];
        let list = null;
        let quote = [];
        const flushParagraph = () => {
          if (!paragraph.length) return;
          const firstLine = paragraph[0].trim();
          const lastLine = paragraph[paragraph.length - 1].trim();
          const isStandaloneVerse = paragraph.length > 1 && /^[“\"]/.test(firstLine) && /[”\"]$/.test(lastLine);
          const rendered = paragraph.map((line) => this.inline(line)).join("<br>");
          html.push(isStandaloneVerse ? `<blockquote class="verse-quote">${rendered}</blockquote>` : `<p>${rendered}</p>`);
          paragraph = [];
        };
        const flushList = () => {
          if (!list) return;
          html.push(`<${list.type}>${list.items.map((item) => `<li>${this.inline(item)}</li>`).join("")}</${list.type}>`);
          list = null;
        };
        const flushQuote = () => {
          if (!quote.length) return;
          html.push(`<blockquote class="verse-quote">${quote.map((line) => this.inline(line)).join("<br>")}</blockquote>`);
          quote = [];
        };
        lines.forEach((rawLine) => {
          const line = rawLine.trimEnd();
          const heading = line.match(/^(#{2,3})\s+(.+)/);
          const ordered = line.match(/^\s*\d+[.)]\s+(.+)/);
          const bullet = line.match(/^\s*[-•]\s+(.+)/);
          if (line.startsWith(">")) {
            flushParagraph(); flushList();
            quote.push(line.replace(/^>\s?/, ""));
          } else if (heading) {
            flushParagraph(); flushList(); flushQuote();
            const level = heading[1].length === 2 ? "h2" : "h3";
            html.push(`<${level}>${this.inline(heading[2])}</${level}>`);
          } else if (ordered || bullet) {
            flushParagraph(); flushQuote();
            const type = ordered ? "ol" : "ul";
            if (list && list.type !== type) flushList();
            if (!list) list = { type, items: [] };
            list.items.push((ordered || bullet)[1]);
          } else if (!line.trim()) {
            flushParagraph(); flushList(); flushQuote();
          } else {
            flushList(); flushQuote();
            paragraph.push(line);
          }
        });
        flushParagraph(); flushList(); flushQuote();
        return html.join("");
      },

      startLoadingStory() {
        let index = 0;
        this.loadingLabel = LOADING_MESSAGES[index];
        window.clearInterval(this.loadingTimer);
        this.loadingTimer = window.setInterval(() => {
          index = (index + 1) % LOADING_MESSAGES.length;
          this.loadingLabel = LOADING_MESSAGES[index];
        }, 2100);
      },

      stopLoadingStory() {
        window.clearInterval(this.loadingTimer);
      },

      scrollBottom(instant) {
        requestAnimationFrame(() => {
          const pageHeight = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
          window.scrollTo({ top: pageHeight, behavior: instant ? "auto" : "smooth" });
          window.setTimeout(() => { this.showJumpToLatest = false; }, instant ? 0 : 300);
        });
      },

      async send() {
        if (this.sending || !this.draft.trim()) return;
        const message = this.draft.trim();
        const firstMessage = !this.hasMessages;
        if (firstMessage) {
          this.recentTitle = this.topicFrom(message);
          const conversation = this.conversations.find((item) => item.id === this.currentConversationId);
          if (conversation) {
            conversation.title = this.recentTitle;
            fetch(`${this.config.conversationDetailBase}${encodeURIComponent(conversation.id)}/`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken() },
              credentials: "same-origin",
              body: JSON.stringify({ title: this.recentTitle })
            }).catch(() => {});
          }
        }
        this.render("user", message);
        this.draft = "";
        this.$nextTick(() => { if (this.$refs.composer) this.$refs.composer.style.height = "auto"; });
        this.scrollBottom(false);
        this.typing = true;
        this.sending = true;
        this.startLoadingStory();
        this.liveStatus = "Kiều Bot đang đọc và soạn phản hồi.";
        const payload = {
          message,
          k: 5,
          model: this.config.model,
          response_length: this.settings.response_length,
          long_answer: this.settings.response_length === "long",
          max_tokens: ({ super_short: 600, short: 1200, long: 1700 })[this.settings.response_length] || 1200
        };
        payload.conversation_id = this.currentConversationId;
        try {
          const response = await fetch(this.config.chatUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrfToken() },
            body: JSON.stringify(payload),
            credentials: "same-origin"
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.message || data.error || "request failed");
          this.typing = false;
          this.render("assistant", data.answer || data.message || "Xin lỗi, yêu cầu chưa được xử lý.", {
            intent: data.intent,
            verification: data.verification,
            elapsed_ms: data.elapsed_ms,
            error: data.error,
            harness: data.harness,
            content_blocks: data.content_blocks
          }, { messageId: data.message_id });
          this.addFollowupsToLast();
          this.liveStatus = "Kiều Bot đã trả lời.";
          this.scrollBottom(false);
        } catch (error) {
          this.typing = false;
          const messageText = error.message === "quota" ? "Bạn đã dùng hết số câu hỏi hôm nay." : "Xin lỗi, có lỗi kỹ thuật khi xử lý câu hỏi.";
          this.render("assistant", messageText);
          this.addFollowupsToLast();
          this.liveStatus = messageText;
          this.scrollBottom(false);
        } finally {
          this.stopLoadingStory();
          this.sending = false;
        }
      }
    };
  };
})();
