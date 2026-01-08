import { useCallback, useMemo } from "react";

type UseAuthOptions = {
  redirectOnUnauthenticated?: boolean;
  redirectPath?: string;
};

// 静态版本的 useAuth hook，不依赖后端 API
// 适用于纯静态部署场景
export function useAuth(options?: UseAuthOptions) {
  const logout = useCallback(async () => {
    // 静态版本不需要实际登出逻辑
    console.log("Static mode: logout called");
  }, []);

  const state = useMemo(() => {
    return {
      user: null,
      loading: false,
      error: null,
      isAuthenticated: false,
    };
  }, []);

  return {
    ...state,
    refresh: () => Promise.resolve(),
    logout,
  };
}
