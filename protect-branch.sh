#!/usr/bin/env bash
###############################################################################
# protect-branch.sh
#
# PRO branch protection + polityka merge dla repo:
#   QHDALabs/QHDALabs-wildfire-risk-pl
#
# Wymaga: gh (GitHub CLI) zalogowane (`gh auth login`) + git.
# Uruchom w Git Bash / WSL / Linux / macOS.
#
# Idempotentny: PUT na endpoint /protection podmienia CALA konfiguracje,
# wiec skrypt mozna odpalac wielokrotnie bez efektow ubocznych.
#
# Wymaga uprawnien ADMIN na repo (branch protection to admin-only operacja).
###############################################################################
set -euo pipefail

# =============================================================================
# KONFIGURACJA  (edytuj tutaj)
# =============================================================================
OWNER="QHDALabs"
REPO="QHDALabs-wildfire-risk-pl"
BRANCH=""                          # puste = auto-detekcja default brancha (main)

# --- Ochrona historii (Twoja glowna prosba) ---------------------------------
ALLOW_FORCE_PUSHES=false           # blokuj force-push
ALLOW_DELETIONS=false              # blokuj kasowanie brancha

# --- Pull Request flow ------------------------------------------------------
REQUIRE_PR_REVIEWS=true            # wymagaj PR (zamiast push wprost na main)
REVIEW_COUNT=1
DISMISS_STALE_REVIEWS=true         # nowy push kasuje stare approvale
REQUIRE_CODE_OWNER_REVIEWS=false   # true TYLKO jesli masz plik CODEOWNERS

# --- Status checks (CI) przed mergem ----------------------------------------
STRICT_STATUS_CHECKS=true          # branch musi byc up-to-date z base
REQUIRED_STATUS_CHECKS=()          # np. ("build" "pytest") — NAZWY jobow z CI.
                                   # PUSTE = bez konkretnych checkow (na razie).
                                   # UWAGA: wpisanie nazwy checka, ktory nie
                                   # istnieje w CI, zablokuje KAZDY merge.

# --- Higiena ----------------------------------------------------------------
REQUIRE_LINEAR_HISTORY=true        # tylko squash/rebase, brak merge-commitow
REQUIRE_CONVERSATION_RESOLUTION=true
REQUIRE_SIGNED_COMMITS=false       # true tylko gdy masz skonfigurowany GPG/SSH signing

# enforce_admins: false = Ty (admin) mozesz w razie potrzeby ominac blokade.
# Dla pracy solo ZOSTAW false, inaczej zablokujesz sie na wlasnym PR
# (GitHub nie pozwala approve'owac wlasnego PR-a).
ENFORCE_ADMINS=false

# --- Polityka merge (ustawienia repo) ---------------------------------------
CONFIGURE_MERGE_POLICY=true
ALLOW_SQUASH_MERGE=true
ALLOW_REBASE_MERGE=true
ALLOW_MERGE_COMMIT=false           # off => spojne z required_linear_history
DELETE_BRANCH_ON_MERGE=true        # auto-sprzatanie branchy po mergu
ALLOW_AUTO_MERGE=true
ALLOW_UPDATE_BRANCH=true

# Tryb testowy: true = tylko pokaz co zrobi, NIC nie wysylaj do GitHuba.
DRY_RUN=false
# =============================================================================

# --- logging ----------------------------------------------------------------
c_reset='\033[0m'; c_red='\033[0;31m'; c_grn='\033[0;32m'; c_yel='\033[0;33m'; c_blu='\033[0;34m'
info(){ printf "${c_blu}==>${c_reset} %s\n" "$*"; }
ok(){   printf "${c_grn}[OK]${c_reset} %s\n" "$*"; }
warn(){ printf "${c_yel}[!]${c_reset} %s\n" "$*" >&2; }
err(){  printf "${c_red}[x]${c_reset} %s\n" "$*" >&2; }
die(){ err "$*"; exit 1; }

# --- preflight --------------------------------------------------------------
preflight(){
  command -v gh  >/dev/null 2>&1 || die "Brak gh (GitHub CLI). Instalacja: https://cli.github.com/"
  command -v git >/dev/null 2>&1 || warn "Brak git w PATH (skrypt zadziala, ale warto miec)."
  gh auth status >/dev/null 2>&1 || die "gh nie jest zalogowane. Uruchom: gh auth login"
}

# --- auto-detekcja default brancha ------------------------------------------
detect_branch(){
  if [ -z "$BRANCH" ]; then
    BRANCH=$(gh api "repos/$OWNER/$REPO" --jq '.default_branch') \
      || die "Nie moge odczytac repo $OWNER/$REPO (uprawnienia? literowka?)."
    info "Default branch wykryty: $BRANCH"
  fi
}

# --- budowa JSON body -------------------------------------------------------
build_body(){
  local contexts_json="[]"
  if [ "${#REQUIRED_STATUS_CHECKS[@]}" -gt 0 ]; then
    contexts_json="[$(printf '"%s",' "${REQUIRED_STATUS_CHECKS[@]}")]"
    contexts_json="${contexts_json/,]/]}"
  fi
  local status_checks_json="{\"strict\": $STRICT_STATUS_CHECKS, \"contexts\": $contexts_json}"

  local pr_reviews_json="null"
  if [ "$REQUIRE_PR_REVIEWS" = true ]; then
    pr_reviews_json="{\"dismiss_stale_reviews\": $DISMISS_STALE_REVIEWS, \
\"require_code_owner_reviews\": $REQUIRE_CODE_OWNER_REVIEWS, \
\"required_approving_review_count\": $REVIEW_COUNT, \
\"require_last_push_approval\": false}"
  fi

  read -r -d '' BODY <<JSON || true
{
  "required_status_checks": $status_checks_json,
  "enforce_admins": $ENFORCE_ADMINS,
  "required_pull_request_reviews": $pr_reviews_json,
  "restrictions": null,
  "required_linear_history": $REQUIRE_LINEAR_HISTORY,
  "allow_force_pushes": $ALLOW_FORCE_PUSHES,
  "allow_deletions": $ALLOW_DELETIONS,
  "block_creations": false,
  "required_conversation_resolution": $REQUIRE_CONVERSATION_RESOLUTION,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON
}

# --- naloz branch protection ------------------------------------------------
apply_protection(){
  if [ "$DRY_RUN" = true ]; then
    info "[DRY_RUN] PUT repos/$OWNER/$REPO/branches/$BRANCH/protection"
    echo "$BODY"
    return 0
  fi
  echo "$BODY" | gh api --method PUT \
    "repos/$OWNER/$REPO/branches/$BRANCH/protection" \
    -H "Accept: application/vnd.github+json" \
    --input - >/dev/null
  ok "Branch protection nalozona na '$BRANCH'."
}

# --- signed commits (osobny endpoint) ---------------------------------------
configure_signatures(){
  if [ "$DRY_RUN" = true ]; then
    info "[DRY_RUN] signatures -> $REQUIRE_SIGNED_COMMITS"
    return 0
  fi
  local method="DELETE"
  [ "$REQUIRE_SIGNED_COMMITS" = true ] && method="POST"
  if gh api --method "$method" \
       "repos/$OWNER/$REPO/branches/$BRANCH/protection/required_signatures" \
       -H "Accept: application/vnd.github+json" >/dev/null 2>&1; then
    ok "Required signatures: $REQUIRE_SIGNED_COMMITS"
  else
    warn "Nie udalo sie ustawic required_signatures (pomijam)."
  fi
}

# --- polityka merge (ustawienia repo) ---------------------------------------
configure_merge_policy(){
  [ "$CONFIGURE_MERGE_POLICY" = true ] || return 0
  if [ "$DRY_RUN" = true ]; then
    info "[DRY_RUN] PATCH repos/$OWNER/$REPO (merge policy)"
    return 0
  fi
  gh api --method PATCH "repos/$OWNER/$REPO" \
    -H "Accept: application/vnd.github+json" \
    -F allow_squash_merge="$ALLOW_SQUASH_MERGE" \
    -F allow_rebase_merge="$ALLOW_REBASE_MERGE" \
    -F allow_merge_commit="$ALLOW_MERGE_COMMIT" \
    -F delete_branch_on_merge="$DELETE_BRANCH_ON_MERGE" \
    -F allow_auto_merge="$ALLOW_AUTO_MERGE" \
    -F allow_update_branch="$ALLOW_UPDATE_BRANCH" >/dev/null
  ok "Polityka merge ustawiona (squash/rebase, auto-delete branch po mergu)."
}

# --- weryfikacja ------------------------------------------------------------
verify(){
  [ "$DRY_RUN" = true ] && return 0
  info "Weryfikacja konfiguracji '$BRANCH':"
  gh api "repos/$OWNER/$REPO/branches/$BRANCH/protection" \
    --jq '{
      force_push_allowed:        .allow_force_pushes.enabled,
      deletions_allowed:         .allow_deletions.enabled,
      linear_history:            .required_linear_history.enabled,
      conversation_resolution:   .required_conversation_resolution.enabled,
      enforce_admins:            .enforce_admins.enabled,
      required_pr_reviews:       (.required_pull_request_reviews.required_approving_review_count // 0),
      strict_status_checks:      (.required_status_checks.strict // false),
      required_contexts:         (.required_status_checks.contexts // [])
    }'
}

# --- main -------------------------------------------------------------------
main(){
  preflight
  info "Repo: $OWNER/$REPO"
  [ "$DRY_RUN" = true ] && warn "DRY_RUN=true — nic nie zostanie wyslane do GitHuba."
  detect_branch
  build_body
  apply_protection
  configure_signatures
  configure_merge_policy
  verify
  ok "Gotowe."
}

main "$@"
