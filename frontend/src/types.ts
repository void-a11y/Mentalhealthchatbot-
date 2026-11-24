export type Route = "answered" | "not_found" | "crisis" | "out_of_scope";

export interface Source {
  n: number;
  title: string;
  section: string;
  source: string;
  snippet: string;
  score: number;
  url: string;
}

export interface ChatResponse {
  response: string;
  route: Route;
  abstained: boolean;
  sources: Source[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  route?: Route;
  sources?: Source[];
}
